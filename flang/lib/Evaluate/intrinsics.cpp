//===-- lib/Evaluate/intrinsics.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Evaluate/intrinsics.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/enum-set.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/common.h"
#include "flang/Evaluate/expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/shape.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/type.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/tools.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <utility>

using namespace Fortran::parser::literals;

namespace Fortran::evaluate {

class FoldingContext;

// This file defines the supported intrinsic procedures and implements
// their recognition and validation.  It is largely table-driven.  See
// docs/intrinsics.md and section 16 of the Fortran 2018 standard
// for full details on each of the intrinsics.  Be advised, they have
// complicated details, and the design of these tables has to accommodate
// that complexity.

// Dummy arguments to generic intrinsic procedures are each specified by
// their keyword name (rarely used, but always defined), allowable type
// categories, a kind pattern, a rank pattern, and information about
// optionality and defaults.  The kind and rank patterns are represented
// here with code values that are significant to the matching/validation engine.

// An actual argument to an intrinsic procedure may be a procedure itself
// only if the dummy argument is Rank::reduceOperation,
// KindCode::addressable, or the special case of NULL(MOLD=procedurePointer).

// These are small bit-sets of type category enumerators.
// Note that typeless (BOZ literal) values don't have a distinct type category.
// These typeless arguments are represented in the tables as if they were
// INTEGER with a special "typeless" kind code.  Arguments of intrinsic types
// that can also be typeless values are encoded with an "elementalOrBOZ"
// rank pattern.
// Assumed-type (TYPE(*)) dummy arguments can be forwarded along to some
// intrinsic functions that accept AnyType + Rank::anyOrAssumedRank or
// AnyType + Kind::addressable.
using CategorySet = common::EnumSet<TypeCategory, 8>;
static constexpr CategorySet IntType{TypeCategory::Integer};
static constexpr CategorySet RealType{TypeCategory::Real};
static constexpr CategorySet ComplexType{TypeCategory::Complex};
static constexpr CategorySet CharType{TypeCategory::Character};
static constexpr CategorySet LogicalType{TypeCategory::Logical};
static constexpr CategorySet IntOrRealType{IntType | RealType};
static constexpr CategorySet IntOrRealOrCharType{IntType | RealType | CharType};
static constexpr CategorySet IntOrLogicalType{IntType | LogicalType};
static constexpr CategorySet FloatingType{RealType | ComplexType};
static constexpr CategorySet NumericType{IntType | RealType | ComplexType};
static constexpr CategorySet RelatableType{IntType | RealType | CharType};
static constexpr CategorySet DerivedType{TypeCategory::Derived};
static constexpr CategorySet IntrinsicType{
    IntType | RealType | ComplexType | CharType | LogicalType};
static constexpr CategorySet AnyType{IntrinsicType | DerivedType};

ENUM_CLASS(KindCode, none, defaultIntegerKind,
    defaultRealKind, // is also the default COMPLEX kind
    doublePrecision, defaultCharKind, defaultLogicalKind,
    any, // matches any kind value; each instance is independent
    // match any kind, but all "same" kinds must be equal. For characters, also
    // implies that lengths must be equal.
    same,
    // for characters that only require the same kind, not length
    sameKind,
    operand, // match any kind, with promotion (non-standard)
    typeless, // BOZ literals are INTEGER with this kind
    teamType, // TEAM_TYPE from module ISO_FORTRAN_ENV (for coarrays)
    kindArg, // this argument is KIND=
    effectiveKind, // for function results: "kindArg" value, possibly defaulted
    dimArg, // this argument is DIM=
    likeMultiply, // for DOT_PRODUCT and MATMUL
    subscript, // address-sized integer
    size, // default KIND= for SIZE(), UBOUND, &c.
    addressable, // for PRESENT(), &c.; anything (incl. procedure) but BOZ
    nullPointerType, // for ASSOCIATED(NULL())
    exactKind, // a single explicit exactKindValue
    atomicIntKind, // atomic_int_kind from iso_fortran_env
    atomicIntOrLogicalKind, // atomic_int_kind or atomic_logical_kind
    sameAtom, // same type and kind as atom
)

struct TypePattern {
  CategorySet categorySet;
  KindCode kindCode{KindCode::none};
  int exactKindValue{0}; // for KindCode::exactKind
  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;
};

// Abbreviations for argument and result patterns in the intrinsic prototypes:

// Match specific kinds of intrinsic types
static constexpr TypePattern DefaultInt{IntType, KindCode::defaultIntegerKind};
static constexpr TypePattern DefaultReal{RealType, KindCode::defaultRealKind};
static constexpr TypePattern DefaultComplex{
    ComplexType, KindCode::defaultRealKind};
static constexpr TypePattern DefaultChar{CharType, KindCode::defaultCharKind};
static constexpr TypePattern DefaultLogical{
    LogicalType, KindCode::defaultLogicalKind};
static constexpr TypePattern BOZ{IntType, KindCode::typeless};
static constexpr TypePattern TeamType{DerivedType, KindCode::teamType};
static constexpr TypePattern DoublePrecision{
    RealType, KindCode::doublePrecision};
static constexpr TypePattern DoublePrecisionComplex{
    ComplexType, KindCode::doublePrecision};
static constexpr TypePattern SubscriptInt{IntType, KindCode::subscript};

// Match any kind of some intrinsic or derived types
static constexpr TypePattern AnyInt{IntType, KindCode::any};
static constexpr TypePattern AnyReal{RealType, KindCode::any};
static constexpr TypePattern AnyIntOrReal{IntOrRealType, KindCode::any};
static constexpr TypePattern AnyIntOrRealOrChar{
    IntOrRealOrCharType, KindCode::any};
static constexpr TypePattern AnyIntOrLogical{IntOrLogicalType, KindCode::any};
static constexpr TypePattern AnyComplex{ComplexType, KindCode::any};
static constexpr TypePattern AnyFloating{FloatingType, KindCode::any};
static constexpr TypePattern AnyNumeric{NumericType, KindCode::any};
static constexpr TypePattern AnyChar{CharType, KindCode::any};
static constexpr TypePattern AnyLogical{LogicalType, KindCode::any};
static constexpr TypePattern AnyRelatable{RelatableType, KindCode::any};
static constexpr TypePattern AnyIntrinsic{IntrinsicType, KindCode::any};
static constexpr TypePattern ExtensibleDerived{DerivedType, KindCode::any};
static constexpr TypePattern AnyData{AnyType, KindCode::any};

// Type is irrelevant, but not BOZ (for PRESENT(), OPTIONAL(), &c.)
static constexpr TypePattern Addressable{AnyType, KindCode::addressable};

// Match some kind of some intrinsic type(s); all "Same" values must match,
// even when not in the same category (e.g., SameComplex and SameReal).
// Can be used to specify a result so long as at least one argument is
// a "Same".
static constexpr TypePattern SameInt{IntType, KindCode::same};
static constexpr TypePattern SameReal{RealType, KindCode::same};
static constexpr TypePattern SameIntOrReal{IntOrRealType, KindCode::same};
static constexpr TypePattern SameComplex{ComplexType, KindCode::same};
static constexpr TypePattern SameFloating{FloatingType, KindCode::same};
static constexpr TypePattern SameNumeric{NumericType, KindCode::same};
static constexpr TypePattern SameChar{CharType, KindCode::same};
static constexpr TypePattern SameCharNoLen{CharType, KindCode::sameKind};
static constexpr TypePattern SameLogical{LogicalType, KindCode::same};
static constexpr TypePattern SameRelatable{RelatableType, KindCode::same};
static constexpr TypePattern SameIntrinsic{IntrinsicType, KindCode::same};
static constexpr TypePattern SameDerivedType{
    CategorySet{TypeCategory::Derived}, KindCode::same};
static constexpr TypePattern SameType{AnyType, KindCode::same};

// Match some kind of some INTEGER or REAL type(s); when argument types
// &/or kinds differ, their values are converted as if they were operands to
// an intrinsic operation like addition.  This is a nonstandard but nearly
// universal extension feature.
static constexpr TypePattern OperandReal{RealType, KindCode::operand};
static constexpr TypePattern OperandIntOrReal{IntOrRealType, KindCode::operand};

// For ASSOCIATED, the first argument is a typeless pointer
static constexpr TypePattern AnyPointer{AnyType, KindCode::nullPointerType};

// For DOT_PRODUCT and MATMUL, the result type depends on the arguments
static constexpr TypePattern ResultLogical{LogicalType, KindCode::likeMultiply};
static constexpr TypePattern ResultNumeric{NumericType, KindCode::likeMultiply};

// Result types with known category and KIND=
static constexpr TypePattern KINDInt{IntType, KindCode::effectiveKind};
static constexpr TypePattern KINDReal{RealType, KindCode::effectiveKind};
static constexpr TypePattern KINDComplex{ComplexType, KindCode::effectiveKind};
static constexpr TypePattern KINDChar{CharType, KindCode::effectiveKind};
static constexpr TypePattern KINDLogical{LogicalType, KindCode::effectiveKind};

static constexpr TypePattern AtomicInt{IntType, KindCode::atomicIntKind};
static constexpr TypePattern AtomicIntOrLogical{
    IntOrLogicalType, KindCode::atomicIntOrLogicalKind};
static constexpr TypePattern SameAtom{IntOrLogicalType, KindCode::sameAtom};

// The default rank pattern for dummy arguments and function results is
// "elemental".
ENUM_CLASS(Rank,
    elemental, // scalar, or array that conforms with other array arguments
    elementalOrBOZ, // elemental, or typeless BOZ literal scalar
    scalar, vector,
    shape, // INTEGER vector of known length and no negative element
    matrix,
    array, // not scalar, rank is known and greater than zero
    coarray, // rank is known and can be scalar; has nonzero corank
    atom, // is scalar and has nonzero corank or is coindexed
    known, // rank is known and can be scalar
    anyOrAssumedRank, // rank can be unknown; assumed-type TYPE(*) allowed
    conformable, // scalar, or array of same rank & shape as "array" argument
    reduceOperation, // a pure function with constraints for REDUCE
    dimReduced, // scalar if no DIM= argument, else rank(array)-1
    dimRemovedOrScalar, // rank(array)-1 (less DIM) or scalar
    scalarIfDim, // scalar if DIM= argument is present, else rank one array
    locReduced, // vector(1:rank) if no DIM= argument, else rank(array)-1
    rankPlus1, // rank(known)+1
    shaped, // rank is length of SHAPE vector
)

ENUM_CLASS(Optionality, required,
    optional, // unless DIM= for SIZE(assumedSize)
    missing, // for DIM= cases like FINDLOC
    repeats, // for MAX/MIN and their several variants
)

ENUM_CLASS(ArgFlag, none,
    canBeNull, // actual argument can be NULL()
    defaultsToSameKind, // for MatchingDefaultKIND
    defaultsToSizeKind, // for SizeDefaultKIND
    defaultsToDefaultForResult, // for DefaultingKIND
)

struct IntrinsicDummyArgument {
  const char *keyword{nullptr};
  TypePattern typePattern;
  Rank rank{Rank::elemental};
  Optionality optionality{Optionality::required};
  common::Intent intent{common::Intent::In};
  common::EnumSet<ArgFlag, 32> flags{};
  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;
};

// constexpr abbreviations for popular arguments:
// DefaultingKIND is a KIND= argument whose default value is the appropriate
// KIND(0), KIND(0.0), KIND(''), &c. value for the function result.
static constexpr IntrinsicDummyArgument DefaultingKIND{"kind",
    {IntType, KindCode::kindArg}, Rank::scalar, Optionality::optional,
    common::Intent::In, {ArgFlag::defaultsToDefaultForResult}};
// MatchingDefaultKIND is a KIND= argument whose default value is the
// kind of any "Same" function argument (viz., the one whose kind pattern is
// "same").
static constexpr IntrinsicDummyArgument MatchingDefaultKIND{"kind",
    {IntType, KindCode::kindArg}, Rank::scalar, Optionality::optional,
    common::Intent::In, {ArgFlag::defaultsToSameKind}};
// SizeDefaultKind is a KIND= argument whose default value should be
// the kind of INTEGER used for address calculations, and can be
// set so with a compiler flag; but the standard mandates the
// kind of default INTEGER.
static constexpr IntrinsicDummyArgument SizeDefaultKIND{"kind",
    {IntType, KindCode::kindArg}, Rank::scalar, Optionality::optional,
    common::Intent::In, {ArgFlag::defaultsToSizeKind}};
static constexpr IntrinsicDummyArgument RequiredDIM{"dim",
    {IntType, KindCode::dimArg}, Rank::scalar, Optionality::required,
    common::Intent::In};
static constexpr IntrinsicDummyArgument OptionalDIM{"dim",
    {IntType, KindCode::dimArg}, Rank::scalar, Optionality::optional,
    common::Intent::In};
static constexpr IntrinsicDummyArgument MissingDIM{"dim",
    {IntType, KindCode::dimArg}, Rank::scalar, Optionality::missing,
    common::Intent::In};
static constexpr IntrinsicDummyArgument OptionalMASK{"mask", AnyLogical,
    Rank::conformable, Optionality::optional, common::Intent::In};
static constexpr IntrinsicDummyArgument OptionalTEAM{
    "team", TeamType, Rank::scalar, Optionality::optional, common::Intent::In};

struct IntrinsicInterface {
  static constexpr int maxArguments{7}; // if not a MAX/MIN(...)
  const char *name{nullptr};
  IntrinsicDummyArgument dummy[maxArguments];
  TypePattern result;
  Rank rank{Rank::elemental};
  IntrinsicClass intrinsicClass{IntrinsicClass::elementalFunction};
  std::optional<SpecificCall> Match(const CallCharacteristics &,
      const common::IntrinsicTypeDefaultKinds &, ActualArguments &,
      FoldingContext &context, const semantics::Scope *builtins) const;
  int CountArguments() const;
  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;
};

int IntrinsicInterface::CountArguments() const {
  int n{0};
  while (n < maxArguments && dummy[n].keyword) {
    ++n;
  }
  return n;
}

// GENERIC INTRINSIC FUNCTION INTERFACES
// Each entry in this table defines a pattern.  Some intrinsic
// functions have more than one such pattern.  Besides the name
// of the intrinsic function, each pattern has specifications for
// the dummy arguments and for the result of the function.
// The dummy argument patterns each have a name (these are from the
// standard, but rarely appear in actual code), a type and kind
// pattern, allowable ranks, and optionality indicators.
// Be advised, the default rank pattern is "elemental".
static const IntrinsicInterface genericIntrinsicFunction[]{
    {"abs", {{"a", SameIntOrReal}}, SameIntOrReal},
    {"abs", {{"a", SameComplex}}, SameReal},
    {"achar", {{"i", AnyInt, Rank::elementalOrBOZ}, DefaultingKIND}, KINDChar},
    {"acos", {{"x", SameFloating}}, SameFloating},
    {"acosd", {{"x", SameFloating}}, SameFloating},
    {"acosh", {{"x", SameFloating}}, SameFloating},
    {"adjustl", {{"string", SameChar}}, SameChar},
    {"adjustr", {{"string", SameChar}}, SameChar},
    {"aimag", {{"z", SameComplex}}, SameReal},
    {"aint", {{"a", SameReal}, MatchingDefaultKIND}, KINDReal},
    {"all", {{"mask", SameLogical, Rank::array}, OptionalDIM}, SameLogical,
        Rank::dimReduced, IntrinsicClass::transformationalFunction},
    {"allocated", {{"array", AnyData, Rank::array}}, DefaultLogical,
        Rank::elemental, IntrinsicClass::inquiryFunction},
    {"allocated", {{"scalar", AnyData, Rank::scalar}}, DefaultLogical,
        Rank::elemental, IntrinsicClass::inquiryFunction},
    {"anint", {{"a", SameReal}, MatchingDefaultKIND}, KINDReal},
    {"any", {{"mask", SameLogical, Rank::array}, OptionalDIM}, SameLogical,
        Rank::dimReduced, IntrinsicClass::transformationalFunction},
    {"asin", {{"x", SameFloating}}, SameFloating},
    {"asind", {{"x", SameFloating}}, SameFloating},
    {"asinh", {{"x", SameFloating}}, SameFloating},
    {"associated",
        {{"pointer", AnyPointer, Rank::known, Optionality::required,
             common::Intent::In, {ArgFlag::canBeNull}},
            {"target", Addressable, Rank::known, Optionality::optional,
                common::Intent::In, {ArgFlag::canBeNull}}},
        DefaultLogical, Rank::elemental, IntrinsicClass::inquiryFunction},
    {"atan", {{"x", SameFloating}}, SameFloating},
    {"atand", {{"x", SameFloating}}, SameFloating},
    {"atan", {{"y", OperandReal}, {"x", OperandReal}}, OperandReal},
    {"atand", {{"y", OperandReal}, {"x", OperandReal}}, OperandReal},
    {"atan2", {{"y", OperandReal}, {"x", OperandReal}}, OperandReal},
    {"atan2d", {{"y", OperandReal}, {"x", OperandReal}}, OperandReal},
    {"atanh", {{"x", SameFloating}}, SameFloating},
    {"bessel_j0", {{"x", SameReal}}, SameReal},
    {"bessel_j1", {{"x", SameReal}}, SameReal},
    {"bessel_jn", {{"n", AnyInt}, {"x", SameReal}}, SameReal},
    {"bessel_jn",
        {{"n1", AnyInt, Rank::scalar}, {"n2", AnyInt, Rank::scalar},
            {"x", SameReal, Rank::scalar}},
        SameReal, Rank::vector, IntrinsicClass::transformationalFunction},
    {"bessel_y0", {{"x", SameReal}}, SameReal},
    {"bessel_y1", {{"x", SameReal}}, SameReal},
    {"bessel_yn", {{"n", AnyInt}, {"x", SameReal}}, SameReal},
    {"bessel_yn",
        {{"n1", AnyInt, Rank::scalar}, {"n2", AnyInt, Rank::scalar},
            {"x", SameReal, Rank::scalar}},
        SameReal, Rank::vector, IntrinsicClass::transformationalFunction},
    {"bge",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DefaultLogical},
    {"bgt",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DefaultLogical},
    {"bit_size",
        {{"i", SameInt, Rank::anyOrAssumedRank, Optionality::required,
            common::Intent::In, {ArgFlag::canBeNull}}},
        SameInt, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"ble",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DefaultLogical},
    {"blt",
        {{"i", AnyInt, Rank::elementalOrBOZ},
            {"j", AnyInt, Rank::elementalOrBOZ}},
        DefaultLogical},
    {"btest", {{"i", AnyInt, Rank::elementalOrBOZ}, {"pos", AnyInt}},
        DefaultLogical},
    {"ceiling", {{"a", AnyReal}, DefaultingKIND}, KINDInt},
    {"char", {{"i", AnyInt, Rank::elementalOrBOZ}, DefaultingKIND}, KINDChar},
    {"cmplx", {{"x", AnyComplex}, DefaultingKIND}, KINDComplex},
    {"cmplx",
        {{"x", AnyIntOrReal, Rank::elementalOrBOZ},
            {"y", AnyIntOrReal, Rank::elementalOrBOZ, Optionality::optional},
            DefaultingKIND},
        KINDComplex},
    {"command_argument_count", {}, DefaultInt, Rank::scalar,
        IntrinsicClass::transformationalFunction},
    {"conjg", {{"z", SameComplex}}, SameComplex},
    {"cos", {{"x", SameFloating}}, SameFloating},
    {"cosd", {{"x", SameFloating}}, SameFloating},
    {"cosh", {{"x", SameFloating}}, SameFloating},
    {"count", {{"mask", AnyLogical, Rank::array}, OptionalDIM, DefaultingKIND},
        KINDInt, Rank::dimReduced, IntrinsicClass::transformationalFunction},
    {"cshift",
        {{"array", SameType, Rank::array},
            {"shift", AnyInt, Rank::dimRemovedOrScalar}, OptionalDIM},
        SameType, Rank::conformable, IntrinsicClass::transformationalFunction},
    {"dble", {{"a", AnyNumeric, Rank::elementalOrBOZ}}, DoublePrecision},
    {"digits",
        {{"x", AnyIntOrReal, Rank::anyOrAssumedRank, Optionality::required,
            common::Intent::In, {ArgFlag::canBeNull}}},
        DefaultInt, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"dim", {{"x", OperandIntOrReal}, {"y", OperandIntOrReal}},
        OperandIntOrReal},
    {"dot_product",
        {{"vector_a", AnyLogical, Rank::vector},
            {"vector_b", AnyLogical, Rank::vector}},
        ResultLogical, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"dot_product",
        {{"vector_a", AnyComplex, Rank::vector},
            {"vector_b", AnyNumeric, Rank::vector}},
        ResultNumeric, Rank::scalar, // conjugates vector_a
        IntrinsicClass::transformationalFunction},
    {"dot_product",
        {{"vector_a", AnyIntOrReal, Rank::vector},
            {"vector_b", AnyNumeric, Rank::vector}},
        ResultNumeric, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"dprod", {{"x", DefaultReal}, {"y", DefaultReal}}, DoublePrecision},
    {"dshiftl",
        {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ},
            {"shift", AnyInt}},
        SameInt},
    {"dshiftl", {{"i", BOZ}, {"j", SameInt}, {"shift", AnyInt}}, SameInt},
    {"dshiftr",
        {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ},
            {"shift", AnyInt}},
        SameInt},
    {"dshiftr", {{"i", BOZ}, {"j", SameInt}, {"shift", AnyInt}}, SameInt},
    {"eoshift",
        {{"array", SameIntrinsic, Rank::array},
            {"shift", AnyInt, Rank::dimRemovedOrScalar},
            {"boundary", SameIntrinsic, Rank::dimRemovedOrScalar,
                Optionality::optional},
            OptionalDIM},
        SameIntrinsic, Rank::conformable,
        IntrinsicClass::transformationalFunction},
    {"eoshift",
        {{"array", SameDerivedType, Rank::array},
            {"shift", AnyInt, Rank::dimRemovedOrScalar},
            // BOUNDARY= is not optional for derived types
            {"boundary", SameDerivedType, Rank::dimRemovedOrScalar},
            OptionalDIM},
        SameDerivedType, Rank::conformable,
        IntrinsicClass::transformationalFunction},
    {"epsilon",
        {{"x", SameReal, Rank::anyOrAssumedRank, Optionality::required,
            common::Intent::In, {ArgFlag::canBeNull}}},
        SameReal, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"erf", {{"x", SameReal}}, SameReal},
    {"erfc", {{"x", SameReal}}, SameReal},
    {"erfc_scaled", {{"x", SameReal}}, SameReal},
    {"exp", {{"x", SameFloating}}, SameFloating},
    {"exp", {{"x", SameFloating}}, SameFloating},
    {"exponent", {{"x", AnyReal}}, DefaultInt},
    {"exp", {{"x", SameFloating}}, SameFloating},
    {"extends_type_of",
        {{"a", ExtensibleDerived, Rank::anyOrAssumedRank},
            {"mold", ExtensibleDerived, Rank::anyOrAssumedRank}},
        DefaultLogical, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"failed_images", {OptionalTEAM, SizeDefaultKIND}, KINDInt, Rank::vector,
        IntrinsicClass::transformationalFunction},
    {"findloc",
        {{"array", AnyNumeric, Rank::array},
            {"value", AnyNumeric, Rank::scalar}, RequiredDIM, OptionalMASK,
            SizeDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::locReduced, IntrinsicClass::transformationalFunction},
    {"findloc",
        {{"array", AnyNumeric, Rank::array},
            {"value", AnyNumeric, Rank::scalar}, MissingDIM, OptionalMASK,
            SizeDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::vector, IntrinsicClass::transformationalFunction},
    {"findloc",
        {{"array", SameCharNoLen, Rank::array},
            {"value", SameCharNoLen, Rank::scalar}, RequiredDIM, OptionalMASK,
            SizeDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::locReduced, IntrinsicClass::transformationalFunction},
    {"findloc",
        {{"array", SameCharNoLen, Rank::array},
            {"value", SameCharNoLen, Rank::scalar}, MissingDIM, OptionalMASK,
            SizeDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::vector, IntrinsicClass::transformationalFunction},
    {"findloc",
        {{"array", AnyLogical, Rank::array},
            {"value", AnyLogical, Rank::scalar}, RequiredDIM, OptionalMASK,
            SizeDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::locReduced, IntrinsicClass::transformationalFunction},
    {"findloc",
        {{"array", AnyLogical, Rank::array},
            {"value", AnyLogical, Rank::scalar}, MissingDIM, OptionalMASK,
            SizeDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::vector, IntrinsicClass::transformationalFunction},
    {"floor", {{"a", AnyReal}, DefaultingKIND}, KINDInt},
    {"fraction", {{"x", SameReal}}, SameReal},
    {"gamma", {{"x", SameReal}}, SameReal},
    {"get_team", {{"level", DefaultInt, Rank::scalar, Optionality::optional}},
        TeamType, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"huge",
        {{"x", SameIntOrReal, Rank::anyOrAssumedRank, Optionality::required,
            common::Intent::In, {ArgFlag::canBeNull}}},
        SameIntOrReal, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"hypot", {{"x", OperandReal}, {"y", OperandReal}}, OperandReal},
    {"iachar", {{"c", AnyChar}, DefaultingKIND}, KINDInt},
    {"iall", {{"array", SameInt, Rank::array}, RequiredDIM, OptionalMASK},
        SameInt, Rank::dimReduced, IntrinsicClass::transformationalFunction},
    {"iall", {{"array", SameInt, Rank::array}, MissingDIM, OptionalMASK},
        SameInt, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"iany", {{"array", SameInt, Rank::array}, RequiredDIM, OptionalMASK},
        SameInt, Rank::dimReduced, IntrinsicClass::transformationalFunction},
    {"iany", {{"array", SameInt, Rank::array}, MissingDIM, OptionalMASK},
        SameInt, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"iparity", {{"array", SameInt, Rank::array}, RequiredDIM, OptionalMASK},
        SameInt, Rank::dimReduced, IntrinsicClass::transformationalFunction},
    {"iparity", {{"array", SameInt, Rank::array}, MissingDIM, OptionalMASK},
        SameInt, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"iand", {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ}}, SameInt},
    {"iand", {{"i", BOZ}, {"j", SameInt}}, SameInt},
    {"ibclr", {{"i", SameInt}, {"pos", AnyInt}}, SameInt},
    {"ibits", {{"i", SameInt}, {"pos", AnyInt}, {"len", AnyInt}}, SameInt},
    {"ibset", {{"i", SameInt}, {"pos", AnyInt}}, SameInt},
    {"ichar", {{"c", AnyChar}, DefaultingKIND}, KINDInt},
    {"ieor", {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ}}, SameInt},
    {"ieor", {{"i", BOZ}, {"j", SameInt}}, SameInt},
    {"image_status", {{"image", SameInt}, OptionalTEAM}, DefaultInt},
    {"index",
        {{"string", SameCharNoLen}, {"substring", SameCharNoLen},
            {"back", AnyLogical, Rank::elemental, Optionality::optional},
            DefaultingKIND},
        KINDInt},
    {"int", {{"a", AnyNumeric, Rank::elementalOrBOZ}, DefaultingKIND}, KINDInt},
    {"int_ptr_kind", {}, DefaultInt, Rank::scalar},
    {"ior", {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ}}, SameInt},
    {"ior", {{"i", BOZ}, {"j", SameInt}}, SameInt},
    {"ishft", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"ishftc",
        {{"i", SameInt}, {"shift", AnyInt},
            {"size", AnyInt, Rank::elemental, Optionality::optional}},
        SameInt},
    {"isnan", {{"a", AnyFloating}}, DefaultLogical},
    {"is_contiguous", {{"array", Addressable, Rank::anyOrAssumedRank}},
        DefaultLogical, Rank::elemental, IntrinsicClass::inquiryFunction},
    {"is_iostat_end", {{"i", AnyInt}}, DefaultLogical},
    {"is_iostat_eor", {{"i", AnyInt}}, DefaultLogical},
    {"izext", {{"i", AnyInt}}, TypePattern{IntType, KindCode::exactKind, 2}},
    {"jzext", {{"i", AnyInt}}, DefaultInt},
    {"kind",
        {{"x", AnyIntrinsic, Rank::elemental, Optionality::required,
            common::Intent::In, {ArgFlag::canBeNull}}},
        DefaultInt, Rank::elemental, IntrinsicClass::inquiryFunction},
    {"lbound",
        {{"array", AnyData, Rank::anyOrAssumedRank}, RequiredDIM,
            SizeDefaultKIND},
        KINDInt, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"lbound", {{"array", AnyData, Rank::anyOrAssumedRank}, SizeDefaultKIND},
        KINDInt, Rank::vector, IntrinsicClass::inquiryFunction},
    {"lcobound",
        {{"coarray", AnyData, Rank::coarray}, OptionalDIM, SizeDefaultKIND},
        KINDInt, Rank::scalarIfDim, IntrinsicClass::inquiryFunction},
    {"leadz", {{"i", AnyInt}}, DefaultInt},
    {"len",
        {{"string", AnyChar, Rank::anyOrAssumedRank, Optionality::required,
             common::Intent::In, {ArgFlag::canBeNull}},
            DefaultingKIND},
        KINDInt, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"len_trim", {{"string", AnyChar}, DefaultingKIND}, KINDInt},
    {"lge", {{"string_a", SameCharNoLen}, {"string_b", SameCharNoLen}},
        DefaultLogical},
    {"lgt", {{"string_a", SameCharNoLen}, {"string_b", SameCharNoLen}},
        DefaultLogical},
    {"lle", {{"string_a", SameCharNoLen}, {"string_b", SameCharNoLen}},
        DefaultLogical},
    {"llt", {{"string_a", SameCharNoLen}, {"string_b", SameCharNoLen}},
        DefaultLogical},
    {"loc", {{"x", Addressable, Rank::anyOrAssumedRank}}, SubscriptInt,
        Rank::scalar},
    {"log", {{"x", SameFloating}}, SameFloating},
    {"log10", {{"x", SameReal}}, SameReal},
    {"logical", {{"l", AnyLogical}, DefaultingKIND}, KINDLogical},
    {"log_gamma", {{"x", SameReal}}, SameReal},
    {"matmul",
        {{"matrix_a", AnyLogical, Rank::vector},
            {"matrix_b", AnyLogical, Rank::matrix}},
        ResultLogical, Rank::vector, IntrinsicClass::transformationalFunction},
    {"matmul",
        {{"matrix_a", AnyLogical, Rank::matrix},
            {"matrix_b", AnyLogical, Rank::vector}},
        ResultLogical, Rank::vector, IntrinsicClass::transformationalFunction},
    {"matmul",
        {{"matrix_a", AnyLogical, Rank::matrix},
            {"matrix_b", AnyLogical, Rank::matrix}},
        ResultLogical, Rank::matrix, IntrinsicClass::transformationalFunction},
    {"matmul",
        {{"matrix_a", AnyNumeric, Rank::vector},
            {"matrix_b", AnyNumeric, Rank::matrix}},
        ResultNumeric, Rank::vector, IntrinsicClass::transformationalFunction},
    {"matmul",
        {{"matrix_a", AnyNumeric, Rank::matrix},
            {"matrix_b", AnyNumeric, Rank::vector}},
        ResultNumeric, Rank::vector, IntrinsicClass::transformationalFunction},
    {"matmul",
        {{"matrix_a", AnyNumeric, Rank::matrix},
            {"matrix_b", AnyNumeric, Rank::matrix}},
        ResultNumeric, Rank::matrix, IntrinsicClass::transformationalFunction},
    {"maskl", {{"i", AnyInt}, DefaultingKIND}, KINDInt},
    {"maskr", {{"i", AnyInt}, DefaultingKIND}, KINDInt},
    {"max",
        {{"a1", OperandIntOrReal}, {"a2", OperandIntOrReal},
            {"a3", OperandIntOrReal, Rank::elemental, Optionality::repeats}},
        OperandIntOrReal},
    {"max",
        {{"a1", SameCharNoLen}, {"a2", SameCharNoLen},
            {"a3", SameCharNoLen, Rank::elemental, Optionality::repeats}},
        SameCharNoLen},
    {"maxexponent",
        {{"x", AnyReal, Rank::anyOrAssumedRank, Optionality::required,
            common::Intent::In, {ArgFlag::canBeNull}}},
        DefaultInt, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"maxloc",
        {{"array", AnyRelatable, Rank::array}, RequiredDIM, OptionalMASK,
            SizeDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::locReduced, IntrinsicClass::transformationalFunction},
    {"maxloc",
        {{"array", AnyRelatable, Rank::array}, MissingDIM, OptionalMASK,
            SizeDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::locReduced, IntrinsicClass::transformationalFunction},
    {"maxval",
        {{"array", SameRelatable, Rank::array}, RequiredDIM, OptionalMASK},
        SameRelatable, Rank::dimReduced,
        IntrinsicClass::transformationalFunction},
    {"maxval",
        {{"array", SameRelatable, Rank::array}, MissingDIM, OptionalMASK},
        SameRelatable, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"merge",
        {{"tsource", SameType}, {"fsource", SameType}, {"mask", AnyLogical}},
        SameType},
    {"merge_bits",
        {{"i", SameInt}, {"j", SameInt, Rank::elementalOrBOZ},
            {"mask", SameInt, Rank::elementalOrBOZ}},
        SameInt},
    {"merge_bits",
        {{"i", BOZ}, {"j", SameInt}, {"mask", SameInt, Rank::elementalOrBOZ}},
        SameInt},
    {"min",
        {{"a1", OperandIntOrReal}, {"a2", OperandIntOrReal},
            {"a3", OperandIntOrReal, Rank::elemental, Optionality::repeats}},
        OperandIntOrReal},
    {"min",
        {{"a1", SameCharNoLen}, {"a2", SameCharNoLen},
            {"a3", SameCharNoLen, Rank::elemental, Optionality::repeats}},
        SameCharNoLen},
    {"minexponent",
        {{"x", AnyReal, Rank::anyOrAssumedRank, Optionality::required,
            common::Intent::In, {ArgFlag::canBeNull}}},
        DefaultInt, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"minloc",
        {{"array", AnyRelatable, Rank::array}, RequiredDIM, OptionalMASK,
            SizeDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::locReduced, IntrinsicClass::transformationalFunction},
    {"minloc",
        {{"array", AnyRelatable, Rank::array}, MissingDIM, OptionalMASK,
            SizeDefaultKIND,
            {"back", AnyLogical, Rank::scalar, Optionality::optional}},
        KINDInt, Rank::locReduced, IntrinsicClass::transformationalFunction},
    {"minval",
        {{"array", SameRelatable, Rank::array}, RequiredDIM, OptionalMASK},
        SameRelatable, Rank::dimReduced,
        IntrinsicClass::transformationalFunction},
    {"minval",
        {{"array", SameRelatable, Rank::array}, MissingDIM, OptionalMASK},
        SameRelatable, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"mod", {{"a", OperandIntOrReal}, {"p", OperandIntOrReal}},
        OperandIntOrReal},
    {"modulo", {{"a", OperandIntOrReal}, {"p", OperandIntOrReal}},
        OperandIntOrReal},
    {"nearest", {{"x", SameReal}, {"s", AnyReal}}, SameReal},
    {"new_line",
        {{"a", SameCharNoLen, Rank::anyOrAssumedRank, Optionality::required,
            common::Intent::In, {ArgFlag::canBeNull}}},
        SameCharNoLen, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"nint", {{"a", AnyReal}, DefaultingKIND}, KINDInt},
    {"norm2", {{"x", SameReal, Rank::array}, OptionalDIM}, SameReal,
        Rank::dimReduced, IntrinsicClass::transformationalFunction},
    {"not", {{"i", SameInt}}, SameInt},
    // NULL() is a special case handled in Probe() below
    {"num_images", {}, DefaultInt, Rank::scalar,
        IntrinsicClass::transformationalFunction},
    {"num_images", {{"team", TeamType, Rank::scalar}}, DefaultInt, Rank::scalar,
        IntrinsicClass::transformationalFunction},
    {"num_images", {{"team_number", AnyInt, Rank::scalar}}, DefaultInt,
        Rank::scalar, IntrinsicClass::transformationalFunction},
    {"out_of_range",
        {{"x", AnyIntOrReal}, {"mold", AnyIntOrReal, Rank::scalar}},
        DefaultLogical},
    {"out_of_range",
        {{"x", AnyReal}, {"mold", AnyInt, Rank::scalar},
            {"round", AnyLogical, Rank::scalar, Optionality::optional}},
        DefaultLogical},
    {"out_of_range", {{"x", AnyReal}, {"mold", AnyReal}}, DefaultLogical},
    {"pack",
        {{"array", SameType, Rank::array},
            {"mask", AnyLogical, Rank::conformable},
            {"vector", SameType, Rank::vector, Optionality::optional}},
        SameType, Rank::vector, IntrinsicClass::transformationalFunction},
    {"parity", {{"mask", SameLogical, Rank::array}, OptionalDIM}, SameLogical,
        Rank::dimReduced, IntrinsicClass::transformationalFunction},
    {"popcnt", {{"i", AnyInt}}, DefaultInt},
    {"poppar", {{"i", AnyInt}}, DefaultInt},
    {"product",
        {{"array", SameNumeric, Rank::array}, RequiredDIM, OptionalMASK},
        SameNumeric, Rank::dimReduced,
        IntrinsicClass::transformationalFunction},
    {"product", {{"array", SameNumeric, Rank::array}, MissingDIM, OptionalMASK},
        SameNumeric, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"precision",
        {{"x", AnyFloating, Rank::anyOrAssumedRank, Optionality::required,
            common::Intent::In, {ArgFlag::canBeNull}}},
        DefaultInt, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"present", {{"a", Addressable, Rank::anyOrAssumedRank}}, DefaultLogical,
        Rank::scalar, IntrinsicClass::inquiryFunction},
    {"radix",
        {{"x", AnyIntOrReal, Rank::anyOrAssumedRank, Optionality::required,
            common::Intent::In, {ArgFlag::canBeNull}}},
        DefaultInt, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"range",
        {{"x", AnyNumeric, Rank::anyOrAssumedRank, Optionality::required,
            common::Intent::In, {ArgFlag::canBeNull}}},
        DefaultInt, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"rank",
        {{"a", AnyData, Rank::anyOrAssumedRank, Optionality::required,
            common::Intent::In, {ArgFlag::canBeNull}}},
        DefaultInt, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"real", {{"a", SameComplex, Rank::elemental}},
        SameReal}, // 16.9.160(4)(ii)
    {"real", {{"a", AnyNumeric, Rank::elementalOrBOZ}, DefaultingKIND},
        KINDReal},
    {"reduce",
        {{"array", SameType, Rank::array},
            {"operation", SameType, Rank::reduceOperation}, RequiredDIM,
            OptionalMASK,
            {"identity", SameType, Rank::scalar, Optionality::optional},
            {"ordered", AnyLogical, Rank::scalar, Optionality::optional}},
        SameType, Rank::dimReduced, IntrinsicClass::transformationalFunction},
    {"reduce",
        {{"array", SameType, Rank::array},
            {"operation", SameType, Rank::reduceOperation}, MissingDIM,
            OptionalMASK,
            {"identity", SameType, Rank::scalar, Optionality::optional},
            {"ordered", AnyLogical, Rank::scalar, Optionality::optional}},
        SameType, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"repeat", {{"string", SameCharNoLen, Rank::scalar}, {"ncopies", AnyInt}},
        SameCharNoLen, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"reshape",
        {{"source", SameType, Rank::array}, {"shape", AnyInt, Rank::shape},
            {"pad", SameType, Rank::array, Optionality::optional},
            {"order", AnyInt, Rank::vector, Optionality::optional}},
        SameType, Rank::shaped, IntrinsicClass::transformationalFunction},
    {"rrspacing", {{"x", SameReal}}, SameReal},
    {"same_type_as",
        {{"a", ExtensibleDerived, Rank::anyOrAssumedRank},
            {"b", ExtensibleDerived, Rank::anyOrAssumedRank}},
        DefaultLogical, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"scale", {{"x", SameReal}, {"i", AnyInt}}, SameReal}, // == IEEE_SCALB()
    {"scan",
        {{"string", SameCharNoLen}, {"set", SameCharNoLen},
            {"back", AnyLogical, Rank::elemental, Optionality::optional},
            DefaultingKIND},
        KINDInt},
    {"selected_char_kind", {{"name", DefaultChar, Rank::scalar}}, DefaultInt,
        Rank::scalar, IntrinsicClass::transformationalFunction},
    {"selected_int_kind", {{"r", AnyInt, Rank::scalar}}, DefaultInt,
        Rank::scalar, IntrinsicClass::transformationalFunction},
    {"selected_real_kind",
        {{"p", AnyInt, Rank::scalar},
            {"r", AnyInt, Rank::scalar, Optionality::optional},
            {"radix", AnyInt, Rank::scalar, Optionality::optional}},
        DefaultInt, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"selected_real_kind",
        {{"p", AnyInt, Rank::scalar, Optionality::optional},
            {"r", AnyInt, Rank::scalar},
            {"radix", AnyInt, Rank::scalar, Optionality::optional}},
        DefaultInt, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"selected_real_kind",
        {{"p", AnyInt, Rank::scalar, Optionality::optional},
            {"r", AnyInt, Rank::scalar, Optionality::optional},
            {"radix", AnyInt, Rank::scalar}},
        DefaultInt, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"set_exponent", {{"x", SameReal}, {"i", AnyInt}}, SameReal},
    {"shape", {{"source", AnyData, Rank::anyOrAssumedRank}, SizeDefaultKIND},
        KINDInt, Rank::vector, IntrinsicClass::inquiryFunction},
    {"shifta", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"shiftl", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"shiftr", {{"i", SameInt}, {"shift", AnyInt}}, SameInt},
    {"sign", {{"a", SameInt}, {"b", AnyInt}}, SameInt},
    {"sign", {{"a", SameReal}, {"b", AnyReal}}, SameReal},
    {"sin", {{"x", SameFloating}}, SameFloating},
    {"sind", {{"x", SameFloating}}, SameFloating},
    {"sinh", {{"x", SameFloating}}, SameFloating},
    {"size",
        {{"array", AnyData, Rank::anyOrAssumedRank},
            OptionalDIM, // unless array is assumed-size
            SizeDefaultKIND},
        KINDInt, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"sizeof", {{"a", AnyData, Rank::anyOrAssumedRank}}, SubscriptInt,
        Rank::scalar, IntrinsicClass::inquiryFunction},
    {"spacing", {{"x", SameReal}}, SameReal},
    {"spread",
        {{"source", SameType, Rank::known}, RequiredDIM,
            {"ncopies", AnyInt, Rank::scalar}},
        SameType, Rank::rankPlus1, IntrinsicClass::transformationalFunction},
    {"sqrt", {{"x", SameFloating}}, SameFloating},
    {"stopped_images", {OptionalTEAM, SizeDefaultKIND}, KINDInt, Rank::vector,
        IntrinsicClass::transformationalFunction},
    {"storage_size",
        {{"a", AnyData, Rank::anyOrAssumedRank, Optionality::required,
             common::Intent::In, {ArgFlag::canBeNull}},
            SizeDefaultKIND},
        KINDInt, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"sum", {{"array", SameNumeric, Rank::array}, RequiredDIM, OptionalMASK},
        SameNumeric, Rank::dimReduced,
        IntrinsicClass::transformationalFunction},
    {"sum", {{"array", SameNumeric, Rank::array}, MissingDIM, OptionalMASK},
        SameNumeric, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"tan", {{"x", SameFloating}}, SameFloating},
    {"tand", {{"x", SameFloating}}, SameFloating},
    {"tanh", {{"x", SameFloating}}, SameFloating},
    {"team_number", {OptionalTEAM}, DefaultInt, Rank::scalar,
        IntrinsicClass::transformationalFunction},
    {"this_image",
        {{"coarray", AnyData, Rank::coarray}, RequiredDIM, OptionalTEAM},
        DefaultInt, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"this_image", {{"coarray", AnyData, Rank::coarray}, OptionalTEAM},
        DefaultInt, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"this_image", {OptionalTEAM}, DefaultInt, Rank::scalar,
        IntrinsicClass::transformationalFunction},
    {"tiny",
        {{"x", SameReal, Rank::anyOrAssumedRank, Optionality::required,
            common::Intent::In, {ArgFlag::canBeNull}}},
        SameReal, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"trailz", {{"i", AnyInt}}, DefaultInt},
    {"transfer",
        {{"source", AnyData, Rank::known}, {"mold", SameType, Rank::scalar}},
        SameType, Rank::scalar, IntrinsicClass::transformationalFunction},
    {"transfer",
        {{"source", AnyData, Rank::known}, {"mold", SameType, Rank::array}},
        SameType, Rank::vector, IntrinsicClass::transformationalFunction},
    {"transfer",
        {{"source", AnyData, Rank::anyOrAssumedRank},
            {"mold", SameType, Rank::anyOrAssumedRank},
            {"size", AnyInt, Rank::scalar}},
        SameType, Rank::vector, IntrinsicClass::transformationalFunction},
    {"transpose", {{"matrix", SameType, Rank::matrix}}, SameType, Rank::matrix,
        IntrinsicClass::transformationalFunction},
    {"trim", {{"string", SameCharNoLen, Rank::scalar}}, SameCharNoLen,
        Rank::scalar, IntrinsicClass::transformationalFunction},
    {"ubound",
        {{"array", AnyData, Rank::anyOrAssumedRank}, RequiredDIM,
            SizeDefaultKIND},
        KINDInt, Rank::scalar, IntrinsicClass::inquiryFunction},
    {"ubound", {{"array", AnyData, Rank::anyOrAssumedRank}, SizeDefaultKIND},
        KINDInt, Rank::vector, IntrinsicClass::inquiryFunction},
    {"ucobound",
        {{"coarray", AnyData, Rank::coarray}, OptionalDIM, SizeDefaultKIND},
        KINDInt, Rank::scalarIfDim, IntrinsicClass::inquiryFunction},
    {"unpack",
        {{"vector", SameType, Rank::vector}, {"mask", AnyLogical, Rank::array},
            {"field", SameType, Rank::conformable}},
        SameType, Rank::conformable, IntrinsicClass::transformationalFunction},
    {"verify",
        {{"string", SameCharNoLen}, {"set", SameCharNoLen},
            {"back", AnyLogical, Rank::elemental, Optionality::optional},
            DefaultingKIND},
        KINDInt},
    {"__builtin_ieee_is_nan", {{"a", AnyFloating}}, DefaultLogical},
    {"__builtin_ieee_is_negative", {{"a", AnyFloating}}, DefaultLogical},
    {"__builtin_ieee_is_normal", {{"a", AnyFloating}}, DefaultLogical},
    {"__builtin_ieee_next_after", {{"x", SameReal}, {"y", AnyReal}}, SameReal},
    {"__builtin_ieee_next_down", {{"x", SameReal}}, SameReal},
    {"__builtin_ieee_next_up", {{"x", SameReal}}, SameReal},
    {"__builtin_ieee_support_datatype",
        {{"x", AnyReal, Rank::elemental, Optionality::optional}},
        DefaultLogical},
    {"__builtin_ieee_support_denormal",
        {{"x", AnyReal, Rank::elemental, Optionality::optional}},
        DefaultLogical},
    {"__builtin_ieee_support_divide",
        {{"x", AnyReal, Rank::elemental, Optionality::optional}},
        DefaultLogical},
    {"__builtin_ieee_support_inf",
        {{"x", AnyReal, Rank::elemental, Optionality::optional}},
        DefaultLogical},
    {"__builtin_ieee_support_io",
        {{"x", AnyReal, Rank::elemental, Optionality::optional}},
        DefaultLogical},
    {"__builtin_ieee_support_nan",
        {{"x", AnyReal, Rank::elemental, Optionality::optional}},
        DefaultLogical},
    {"__builtin_ieee_support_sqrt",
        {{"x", AnyReal, Rank::elemental, Optionality::optional}},
        DefaultLogical},
    {"__builtin_ieee_support_standard",
        {{"x", AnyReal, Rank::elemental, Optionality::optional}},
        DefaultLogical},
    {"__builtin_ieee_support_subnormal",
        {{"x", AnyReal, Rank::elemental, Optionality::optional}},
        DefaultLogical},
    {"__builtin_ieee_support_underflow_control",
        {{"x", AnyReal, Rank::elemental, Optionality::optional}},
        DefaultLogical},
};

// TODO: Coarray intrinsic functions
//  IMAGE_INDEX, COSHAPE
// TODO: Non-standard intrinsic functions
//  SHIFT,
//  COMPL, EQV, NEQV, INT8, JINT, JNINT, KNINT,
//  QCMPLX, QEXT, QFLOAT, QREAL, DNUM,
//  INUM, JNUM, KNUM, QNUM, RNUM, RAN, RANF, ILEN,
//  MCLOCK, SECNDS, COTAN, IBCHNG, ISHA, ISHC, ISHL, IXOR
//  IARG, IARGC, NARGS, NUMARG, BADDRESS, IADDR, CACHESIZE,
//  EOF, FP_CLASS, INT_PTR_KIND, MALLOC
//  probably more (these are PGI + Intel, possibly incomplete)
// TODO: Optionally warn on use of non-standard intrinsics:
//  LOC, probably others
// TODO: Optionally warn on operand promotion extension

// Aliases for a few generic intrinsic functions for legacy
// compatibility and builtins.
static const std::pair<const char *, const char *> genericAlias[]{
    {"and", "iand"},
    {"imag", "aimag"},
    {"lshift", "shiftl"},
    {"or", "ior"},
    {"rshift", "shifta"},
    {"xor", "ieor"},
    {"__builtin_ieee_selected_real_kind", "selected_real_kind"},
};

// The following table contains the intrinsic functions listed in
// Tables 16.2 and 16.3 in Fortran 2018.  The "unrestricted" functions
// in Table 16.2 can be used as actual arguments, PROCEDURE() interfaces,
// and procedure pointer targets.
// Note that the restricted conversion functions dcmplx, dreal, float, idint,
// ifix, and sngl are extended to accept any argument kind because this is a
// common Fortran compilers behavior, and as far as we can tell, is safe and
// useful.
struct SpecificIntrinsicInterface : public IntrinsicInterface {
  const char *generic{nullptr};
  bool isRestrictedSpecific{false};
  // Exact actual/dummy type matching is required by default for specific
  // intrinsics. If useGenericAndForceResultType is set, then the probing will
  // also attempt to use the related generic intrinsic and to convert the result
  // to the specific intrinsic result type if needed. This also prevents
  // using the generic name so that folding can insert the conversion on the
  // result and not the arguments.
  //
  // This is not enabled on all specific intrinsics because an alternative
  // is to convert the actual arguments to the required dummy types and this is
  // not numerically equivalent.
  //  e.g. IABS(INT(i, 4)) not equiv to INT(ABS(i), 4).
  // This is allowed for restricted min/max specific functions because
  // the expected behavior is clear from their definitions. A warning is though
  // always emitted because other compilers' behavior is not ubiquitous here and
  // the results in case of conversion overflow might not be equivalent.
  // e.g for MIN0: INT(MIN(2147483647_8, 2*2147483647_8), 4) = 2147483647_4
  // but: MIN(INT(2147483647_8, 4), INT(2*2147483647_8, 4)) = -2_4
  // xlf and ifort return the first, and pgfortran the later. f18 will return
  // the first because this matches more closely the MIN0 definition in
  // Fortran 2018 table 16.3 (although it is still an extension to allow
  // non default integer argument in MIN0).
  bool useGenericAndForceResultType{false};
};

static const SpecificIntrinsicInterface specificIntrinsicFunction[]{
    {{"abs", {{"a", DefaultReal}}, DefaultReal}},
    {{"acos", {{"x", DefaultReal}}, DefaultReal}},
    {{"aimag", {{"z", DefaultComplex}}, DefaultReal}},
    {{"aint", {{"a", DefaultReal}}, DefaultReal}},
    {{"alog", {{"x", DefaultReal}}, DefaultReal}, "log"},
    {{"alog10", {{"x", DefaultReal}}, DefaultReal}, "log10"},
    {{"amax0",
         {{"a1", DefaultInt}, {"a2", DefaultInt},
             {"a3", DefaultInt, Rank::elemental, Optionality::repeats}},
         DefaultReal},
        "max", true, true},
    {{"amax1",
         {{"a1", DefaultReal}, {"a2", DefaultReal},
             {"a3", DefaultReal, Rank::elemental, Optionality::repeats}},
         DefaultReal},
        "max", true, true},
    {{"amin0",
         {{"a1", DefaultInt}, {"a2", DefaultInt},
             {"a3", DefaultInt, Rank::elemental, Optionality::repeats}},
         DefaultReal},
        "min", true, true},
    {{"amin1",
         {{"a1", DefaultReal}, {"a2", DefaultReal},
             {"a3", DefaultReal, Rank::elemental, Optionality::repeats}},
         DefaultReal},
        "min", true, true},
    {{"amod", {{"a", DefaultReal}, {"p", DefaultReal}}, DefaultReal}, "mod"},
    {{"anint", {{"a", DefaultReal}}, DefaultReal}},
    {{"asin", {{"x", DefaultReal}}, DefaultReal}},
    {{"atan", {{"x", DefaultReal}}, DefaultReal}},
    {{"atan2", {{"y", DefaultReal}, {"x", DefaultReal}}, DefaultReal}},
    {{"babs", {{"a", TypePattern{IntType, KindCode::exactKind, 1}}},
         TypePattern{IntType, KindCode::exactKind, 1}},
        "abs"},
    {{"cabs", {{"a", DefaultComplex}}, DefaultReal}, "abs"},
    {{"ccos", {{"x", DefaultComplex}}, DefaultComplex}, "cos"},
    {{"cdabs", {{"a", DoublePrecisionComplex}}, DoublePrecision}, "abs"},
    {{"cdcos", {{"x", DoublePrecisionComplex}}, DoublePrecisionComplex}, "cos"},
    {{"cdexp", {{"x", DoublePrecisionComplex}}, DoublePrecisionComplex}, "exp"},
    {{"cdlog", {{"x", DoublePrecisionComplex}}, DoublePrecisionComplex}, "log"},
    {{"cdsin", {{"x", DoublePrecisionComplex}}, DoublePrecisionComplex}, "sin"},
    {{"cdsqrt", {{"x", DoublePrecisionComplex}}, DoublePrecisionComplex},
        "sqrt"},
    {{"cexp", {{"x", DefaultComplex}}, DefaultComplex}, "exp"},
    {{"clog", {{"x", DefaultComplex}}, DefaultComplex}, "log"},
    {{"conjg", {{"z", DefaultComplex}}, DefaultComplex}},
    {{"cos", {{"x", DefaultReal}}, DefaultReal}},
    {{"cosh", {{"x", DefaultReal}}, DefaultReal}},
    {{"csin", {{"x", DefaultComplex}}, DefaultComplex}, "sin"},
    {{"csqrt", {{"x", DefaultComplex}}, DefaultComplex}, "sqrt"},
    {{"ctan", {{"x", DefaultComplex}}, DefaultComplex}, "tan"},
    {{"dabs", {{"a", DoublePrecision}}, DoublePrecision}, "abs"},
    {{"dacos", {{"x", DoublePrecision}}, DoublePrecision}, "acos"},
    {{"dasin", {{"x", DoublePrecision}}, DoublePrecision}, "asin"},
    {{"datan", {{"x", DoublePrecision}}, DoublePrecision}, "atan"},
    {{"datan2", {{"y", DoublePrecision}, {"x", DoublePrecision}},
         DoublePrecision},
        "atan2"},
    {{"dcmplx", {{"x", AnyComplex}}, DoublePrecisionComplex}, "cmplx", true},
    {{"dcmplx",
         {{"x", AnyIntOrReal, Rank::elementalOrBOZ},
             {"y", AnyIntOrReal, Rank::elementalOrBOZ, Optionality::optional}},
         DoublePrecisionComplex},
        "cmplx", true},
    {{"dconjg", {{"z", DoublePrecisionComplex}}, DoublePrecisionComplex},
        "conjg"},
    {{"dcos", {{"x", DoublePrecision}}, DoublePrecision}, "cos"},
    {{"dcosh", {{"x", DoublePrecision}}, DoublePrecision}, "cosh"},
    {{"ddim", {{"x", DoublePrecision}, {"y", DoublePrecision}},
         DoublePrecision},
        "dim"},
    {{"dexp", {{"x", DoublePrecision}}, DoublePrecision}, "exp"},
    {{"dfloat", {{"a", AnyInt}}, DoublePrecision}, "real", true},
    {{"dim", {{"x", DefaultReal}, {"y", DefaultReal}}, DefaultReal}},
    {{"dimag", {{"z", DoublePrecisionComplex}}, DoublePrecision}, "aimag"},
    {{"dint", {{"a", DoublePrecision}}, DoublePrecision}, "aint"},
    {{"dlog", {{"x", DoublePrecision}}, DoublePrecision}, "log"},
    {{"dlog10", {{"x", DoublePrecision}}, DoublePrecision}, "log10"},
    {{"dmax1",
         {{"a1", DoublePrecision}, {"a2", DoublePrecision},
             {"a3", DoublePrecision, Rank::elemental, Optionality::repeats}},
         DoublePrecision},
        "max", true, true},
    {{"dmin1",
         {{"a1", DoublePrecision}, {"a2", DoublePrecision},
             {"a3", DoublePrecision, Rank::elemental, Optionality::repeats}},
         DoublePrecision},
        "min", true, true},
    {{"dmod", {{"a", DoublePrecision}, {"p", DoublePrecision}},
         DoublePrecision},
        "mod"},
    {{"dnint", {{"a", DoublePrecision}}, DoublePrecision}, "anint"},
    {{"dprod", {{"x", DefaultReal}, {"y", DefaultReal}}, DoublePrecision}},
    {{"dreal", {{"a", AnyComplex}}, DoublePrecision}, "real", true},
    {{"dsign", {{"a", DoublePrecision}, {"b", DoublePrecision}},
         DoublePrecision},
        "sign"},
    {{"dsin", {{"x", DoublePrecision}}, DoublePrecision}, "sin"},
    {{"dsinh", {{"x", DoublePrecision}}, DoublePrecision}, "sinh"},
    {{"dsqrt", {{"x", DoublePrecision}}, DoublePrecision}, "sqrt"},
    {{"dtan", {{"x", DoublePrecision}}, DoublePrecision}, "tan"},
    {{"dtanh", {{"x", DoublePrecision}}, DoublePrecision}, "tanh"},
    {{"exp", {{"x", DefaultReal}}, DefaultReal}},
    {{"float", {{"a", AnyInt}}, DefaultReal}, "real", true},
    {{"iabs", {{"a", DefaultInt}}, DefaultInt}, "abs"},
    {{"idim", {{"x", DefaultInt}, {"y", DefaultInt}}, DefaultInt}, "dim"},
    {{"idint", {{"a", AnyReal}}, DefaultInt}, "int", true},
    {{"idnint", {{"a", DoublePrecision}}, DefaultInt}, "nint"},
    {{"ifix", {{"a", AnyReal}}, DefaultInt}, "int", true},
    {{"iiabs", {{"a", TypePattern{IntType, KindCode::exactKind, 2}}},
         TypePattern{IntType, KindCode::exactKind, 2}},
        "abs"},
    {{"index", {{"string", DefaultChar}, {"substring", DefaultChar}},
        DefaultInt}},
    {{"isign", {{"a", DefaultInt}, {"b", DefaultInt}}, DefaultInt}, "sign"},
    {{"jiabs", {{"a", TypePattern{IntType, KindCode::exactKind, 4}}},
         TypePattern{IntType, KindCode::exactKind, 4}},
        "abs"},
    {{"kiabs", {{"a", TypePattern{IntType, KindCode::exactKind, 8}}},
         TypePattern{IntType, KindCode::exactKind, 8}},
        "abs"},
    {{"len", {{"string", DefaultChar, Rank::anyOrAssumedRank}}, DefaultInt,
        Rank::scalar, IntrinsicClass::inquiryFunction}},
    {{"lge", {{"string_a", DefaultChar}, {"string_b", DefaultChar}},
         DefaultLogical},
        "lge", true},
    {{"lgt", {{"string_a", DefaultChar}, {"string_b", DefaultChar}},
         DefaultLogical},
        "lgt", true},
    {{"lle", {{"string_a", DefaultChar}, {"string_b", DefaultChar}},
         DefaultLogical},
        "lle", true},
    {{"llt", {{"string_a", DefaultChar}, {"string_b", DefaultChar}},
         DefaultLogical},
        "llt", true},
    {{"log", {{"x", DefaultReal}}, DefaultReal}},
    {{"log10", {{"x", DefaultReal}}, DefaultReal}},
    {{"max0",
         {{"a1", DefaultInt}, {"a2", DefaultInt},
             {"a3", DefaultInt, Rank::elemental, Optionality::repeats}},
         DefaultInt},
        "max", true, true},
    {{"max1",
         {{"a1", DefaultReal}, {"a2", DefaultReal},
             {"a3", DefaultReal, Rank::elemental, Optionality::repeats}},
         DefaultInt},
        "max", true, true},
    {{"min0",
         {{"a1", DefaultInt}, {"a2", DefaultInt},
             {"a3", DefaultInt, Rank::elemental, Optionality::repeats}},
         DefaultInt},
        "min", true, true},
    {{"min1",
         {{"a1", DefaultReal}, {"a2", DefaultReal},
             {"a3", DefaultReal, Rank::elemental, Optionality::repeats}},
         DefaultInt},
        "min", true, true},
    {{"mod", {{"a", DefaultInt}, {"p", DefaultInt}}, DefaultInt}},
    {{"nint", {{"a", DefaultReal}}, DefaultInt}},
    {{"sign", {{"a", DefaultReal}, {"b", DefaultReal}}, DefaultReal}},
    {{"sin", {{"x", DefaultReal}}, DefaultReal}},
    {{"sinh", {{"x", DefaultReal}}, DefaultReal}},
    {{"sngl", {{"a", AnyReal}}, DefaultReal}, "real", true},
    {{"sqrt", {{"x", DefaultReal}}, DefaultReal}},
    {{"tan", {{"x", DefaultReal}}, DefaultReal}},
    {{"tanh", {{"x", DefaultReal}}, DefaultReal}},
    {{"zabs", {{"a", TypePattern{ComplexType, KindCode::exactKind, 8}}},
         TypePattern{RealType, KindCode::exactKind, 8}},
        "abs"},
};

static const IntrinsicInterface intrinsicSubroutine[]{
    {"abort", {}, {}, Rank::elemental, IntrinsicClass::impureSubroutine},
    {"atomic_and",
        {{"atom", AtomicInt, Rank::atom, Optionality::required,
             common::Intent::InOut},
            {"value", AnyInt, Rank::scalar, Optionality::required,
                common::Intent::In},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::atomicSubroutine},
    {"atomic_cas",
        {{"atom", SameAtom, Rank::atom, Optionality::required,
             common::Intent::InOut},
            {"old", SameAtom, Rank::scalar, Optionality::required,
                common::Intent::Out},
            {"compare", SameAtom, Rank::scalar, Optionality::required,
                common::Intent::In},
            {"new", SameAtom, Rank::scalar, Optionality::required,
                common::Intent::In},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::atomicSubroutine},
    {"atomic_define",
        {{"atom", AtomicIntOrLogical, Rank::atom, Optionality::required,
             common::Intent::Out},
            {"value", AnyIntOrLogical, Rank::scalar, Optionality::required,
                common::Intent::In},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::atomicSubroutine},
    {"atomic_fetch_add",
        {{"atom", AtomicInt, Rank::atom, Optionality::required,
             common::Intent::InOut},
            {"value", AnyInt, Rank::scalar, Optionality::required,
                common::Intent::In},
            {"old", AtomicInt, Rank::scalar, Optionality::required,
                common::Intent::Out},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::atomicSubroutine},
    {"atomic_fetch_and",
        {{"atom", AtomicInt, Rank::atom, Optionality::required,
             common::Intent::InOut},
            {"value", AnyInt, Rank::scalar, Optionality::required,
                common::Intent::In},
            {"old", AtomicInt, Rank::scalar, Optionality::required,
                common::Intent::Out},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::atomicSubroutine},
    {"atomic_fetch_or",
        {{"atom", AtomicInt, Rank::atom, Optionality::required,
             common::Intent::InOut},
            {"value", AnyInt, Rank::scalar, Optionality::required,
                common::Intent::In},
            {"old", AtomicInt, Rank::scalar, Optionality::required,
                common::Intent::Out},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::atomicSubroutine},
    {"atomic_fetch_xor",
        {{"atom", AtomicInt, Rank::atom, Optionality::required,
             common::Intent::InOut},
            {"value", AnyInt, Rank::scalar, Optionality::required,
                common::Intent::In},
            {"old", AtomicInt, Rank::scalar, Optionality::required,
                common::Intent::Out},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::atomicSubroutine},
    {"atomic_or",
        {{"atom", AtomicInt, Rank::atom, Optionality::required,
             common::Intent::InOut},
            {"value", AnyInt, Rank::scalar, Optionality::required,
                common::Intent::In},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::atomicSubroutine},
    {"atomic_ref",
        {{"value", AnyIntOrLogical, Rank::scalar, Optionality::required,
             common::Intent::Out},
            {"atom", AtomicIntOrLogical, Rank::atom, Optionality::required,
                common::Intent::In},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::atomicSubroutine},
    {"atomic_xor",
        {{"atom", AtomicInt, Rank::atom, Optionality::required,
             common::Intent::InOut},
            {"value", AnyInt, Rank::scalar, Optionality::required,
                common::Intent::In},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::atomicSubroutine},
    {"co_broadcast",
        {{"a", AnyData, Rank::anyOrAssumedRank, Optionality::required,
             common::Intent::InOut},
            {"source_image", AnyInt, Rank::scalar, Optionality::required,
                common::Intent::In},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"errmsg", DefaultChar, Rank::scalar, Optionality::optional,
                common::Intent::InOut}},
        {}, Rank::elemental, IntrinsicClass::collectiveSubroutine},
    {"co_max",
        {{"a", AnyIntOrRealOrChar, Rank::anyOrAssumedRank,
             Optionality::required, common::Intent::InOut},
            {"result_image", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::In},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"errmsg", DefaultChar, Rank::scalar, Optionality::optional,
                common::Intent::InOut}},
        {}, Rank::elemental, IntrinsicClass::collectiveSubroutine},
    {"co_min",
        {{"a", AnyIntOrRealOrChar, Rank::anyOrAssumedRank,
             Optionality::required, common::Intent::InOut},
            {"result_image", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::In},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"errmsg", DefaultChar, Rank::scalar, Optionality::optional,
                common::Intent::InOut}},
        {}, Rank::elemental, IntrinsicClass::collectiveSubroutine},
    {"co_sum",
        {{"a", AnyNumeric, Rank::anyOrAssumedRank, Optionality::required,
             common::Intent::InOut},
            {"result_image", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::In},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"errmsg", DefaultChar, Rank::scalar, Optionality::optional,
                common::Intent::InOut}},
        {}, Rank::elemental, IntrinsicClass::collectiveSubroutine},
    {"cpu_time",
        {{"time", AnyReal, Rank::scalar, Optionality::required,
            common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::impureSubroutine},
    {"date_and_time",
        {{"date", DefaultChar, Rank::scalar, Optionality::optional,
             common::Intent::Out},
            {"time", DefaultChar, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"zone", DefaultChar, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"values", AnyInt, Rank::vector, Optionality::optional,
                common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::impureSubroutine},
    {"execute_command_line",
        {{"command", DefaultChar, Rank::scalar},
            {"wait", AnyLogical, Rank::scalar, Optionality::optional},
            {"exitstat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::InOut},
            {"cmdstat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"cmdmsg", DefaultChar, Rank::scalar, Optionality::optional,
                common::Intent::InOut}},
        {}, Rank::elemental, IntrinsicClass::impureSubroutine},
    {"exit", {{"status", DefaultInt, Rank::scalar, Optionality::optional}}, {},
        Rank::elemental, IntrinsicClass::impureSubroutine},
    {"get_command",
        {{"command", DefaultChar, Rank::scalar, Optionality::optional,
             common::Intent::Out},
            {"length", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"status", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"errmsg", DefaultChar, Rank::scalar, Optionality::optional,
                common::Intent::InOut}},
        {}, Rank::elemental, IntrinsicClass::impureSubroutine},
    {"get_command_argument",
        {{"number", AnyInt, Rank::scalar},
            {"value", DefaultChar, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"length", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"status", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"errmsg", DefaultChar, Rank::scalar, Optionality::optional,
                common::Intent::InOut}},
        {}, Rank::elemental, IntrinsicClass::impureSubroutine},
    {"get_environment_variable",
        {{"name", DefaultChar, Rank::scalar},
            {"value", DefaultChar, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"length", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"status", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"trim_name", AnyLogical, Rank::scalar, Optionality::optional},
            {"errmsg", DefaultChar, Rank::scalar, Optionality::optional,
                common::Intent::InOut}},
        {}, Rank::elemental, IntrinsicClass::impureSubroutine},
    {"move_alloc",
        {{"from", SameType, Rank::known, Optionality::required,
             common::Intent::InOut},
            {"to", SameType, Rank::known, Optionality::required,
                common::Intent::Out},
            {"stat", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"errmsg", DefaultChar, Rank::scalar, Optionality::optional,
                common::Intent::InOut}},
        {}, Rank::elemental, IntrinsicClass::pureSubroutine},
    {"mvbits",
        {{"from", SameInt}, {"frompos", AnyInt}, {"len", AnyInt},
            {"to", SameInt, Rank::elemental, Optionality::required,
                common::Intent::Out},
            {"topos", AnyInt}},
        {}, Rank::elemental, IntrinsicClass::elementalSubroutine}, // elemental
    {"random_init",
        {{"repeatable", AnyLogical, Rank::scalar},
            {"image_distinct", AnyLogical, Rank::scalar}},
        {}, Rank::elemental, IntrinsicClass::impureSubroutine},
    {"random_number",
        {{"harvest", AnyReal, Rank::known, Optionality::required,
            common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::impureSubroutine},
    {"random_seed",
        {{"size", DefaultInt, Rank::scalar, Optionality::optional,
             common::Intent::Out},
            {"put", DefaultInt, Rank::vector, Optionality::optional},
            {"get", DefaultInt, Rank::vector, Optionality::optional,
                common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::impureSubroutine},
    {"system_clock",
        {{"count", AnyInt, Rank::scalar, Optionality::optional,
             common::Intent::Out},
            {"count_rate", AnyIntOrReal, Rank::scalar, Optionality::optional,
                common::Intent::Out},
            {"count_max", AnyInt, Rank::scalar, Optionality::optional,
                common::Intent::Out}},
        {}, Rank::elemental, IntrinsicClass::impureSubroutine},
};

// TODO: Intrinsic subroutine EVENT_QUERY
// TODO: Atomic intrinsic subroutines: ATOMIC_ADD
// TODO: Collective intrinsic subroutines: co_reduce

// Finds a built-in derived type and returns it as a DynamicType.
static DynamicType GetBuiltinDerivedType(
    const semantics::Scope *builtinsScope, const char *which) {
  if (!builtinsScope) {
    common::die("INTERNAL: The __fortran_builtins module was not found, and "
                "the type '%s' was required",
        which);
  }
  auto iter{
      builtinsScope->find(semantics::SourceName{which, std::strlen(which)})};
  if (iter == builtinsScope->cend()) {
    common::die(
        "INTERNAL: The __fortran_builtins module does not define the type '%s'",
        which);
  }
  const semantics::Symbol &symbol{*iter->second};
  const semantics::Scope &scope{DEREF(symbol.scope())};
  const semantics::DerivedTypeSpec &derived{DEREF(scope.derivedTypeSpec())};
  return DynamicType{derived};
}

static std::int64_t GetBuiltinKind(
    const semantics::Scope *builtinsScope, const char *which) {
  if (!builtinsScope) {
    common::die("INTERNAL: The __fortran_builtins module was not found, and "
                "the kind '%s' was required",
        which);
  }
  auto iter{
      builtinsScope->find(semantics::SourceName{which, std::strlen(which)})};
  if (iter == builtinsScope->cend()) {
    common::die(
        "INTERNAL: The __fortran_builtins module does not define the kind '%s'",
        which);
  }
  const semantics::Symbol &symbol{*iter->second};
  const auto &details{
      DEREF(symbol.detailsIf<semantics::ObjectEntityDetails>())};
  if (const auto kind{ToInt64(details.init())}) {
    return *kind;
  } else {
    common::die(
        "INTERNAL: The __fortran_builtins module does not define the kind '%s'",
        which);
    return -1;
  }
}

// Ensure that the keywords of arguments to MAX/MIN and their variants
// are of the form A123 with no duplicates or leading zeroes.
static bool CheckMaxMinArgument(std::optional<parser::CharBlock> keyword,
    std::set<parser::CharBlock> &set, const char *intrinsicName,
    parser::ContextualMessages &messages) {
  if (keyword) {
    std::size_t j{1};
    for (; j < keyword->size(); ++j) {
      char ch{(*keyword)[j]};
      if (ch < (j == 1 ? '1' : '0') || ch > '9') {
        break;
      }
    }
    if (keyword->size() < 2 || (*keyword)[0] != 'a' || j < keyword->size()) {
      messages.Say(*keyword,
          "Argument keyword '%s=' is not known in call to '%s'"_err_en_US,
          *keyword, intrinsicName);
      return false;
    }
    auto [_, wasInserted]{set.insert(*keyword)};
    if (!wasInserted) {
      messages.Say(*keyword,
          "Argument keyword '%s=' was repeated in call to '%s'"_err_en_US,
          *keyword, intrinsicName);
      return false;
    }
  }
  return true;
}

static void CheckMaxMinA1A2Argument(const ActualArguments &arguments,
    std::set<parser::CharBlock> &set, parser::ContextualMessages &messages) {
  parser::CharBlock kwA1{"a1", 2};
  parser::CharBlock kwA2{"a2", 2};
  bool missingA1{set.find(kwA1) == set.end()};
  bool missingA2{set.find(kwA2) == set.end()};

  if (arguments.size() > 1) {
    if (arguments.at(0)->keyword()) {
      // If the keyword is specified in the first argument, the following
      // arguments must have the keywords.
      if (missingA1 && missingA2) {
        messages.Say("missing mandatory '%s=' and '%s=' arguments"_err_en_US,
            kwA1.ToString(), kwA2.ToString());
      } else if (missingA1 && !missingA2) {
        messages.Say(
            "missing mandatory '%s=' argument"_err_en_US, kwA1.ToString());
      } else if (!missingA1 && missingA2) {
        messages.Say(
            "missing mandatory '%s=' argument"_err_en_US, kwA2.ToString());
      }
    } else if (arguments.at(1)->keyword()) {
      // No keyword is specified in the first argument.
      if (missingA1 && missingA2) {
        messages.Say(
            "missing mandatory '%s=' argument"_err_en_US, kwA2.ToString());
      }
    }
  }
}

static bool CheckAtomicKind(const ActualArgument &arg,
    const semantics::Scope *builtinsScope,
    parser::ContextualMessages &messages) {
  std::string atomicKindStr;
  std::optional<DynamicType> type{arg.GetType()};

  if (type->category() == TypeCategory::Integer) {
    atomicKindStr = "atomic_int_kind";
  } else if (type->category() == TypeCategory::Logical) {
    atomicKindStr = "atomic_logical_kind";
  } else {
    common::die("atomic_int_kind or atomic_logical_kind from iso_fortran_env "
                "must be used with IntType or LogicalType");
  }

  bool argOk = type->kind() ==
      GetBuiltinKind(builtinsScope, ("__builtin_" + atomicKindStr).c_str());
  if (!argOk) {
    messages.Say(arg.sourceLocation(),
        "Actual argument for 'atom=' must have kind=atomic_int_kind or atomic_logical_kind, but is '%s'"_err_en_US,
        type->AsFortran());
  }
  return argOk;
}

// Intrinsic interface matching against the arguments of a particular
// procedure reference.
std::optional<SpecificCall> IntrinsicInterface::Match(
    const CallCharacteristics &call,
    const common::IntrinsicTypeDefaultKinds &defaults,
    ActualArguments &arguments, FoldingContext &context,
    const semantics::Scope *builtinsScope) const {
  auto &messages{context.messages()};
  // Attempt to construct a 1-1 correspondence between the dummy arguments in
  // a particular intrinsic procedure's generic interface and the actual
  // arguments in a procedure reference.
  std::size_t dummyArgPatterns{0};
  for (; dummyArgPatterns < maxArguments && dummy[dummyArgPatterns].keyword;
       ++dummyArgPatterns) {
  }
  // MAX and MIN (and others that map to them) allow their last argument to
  // be repeated indefinitely.  The actualForDummy vector is sized
  // and null-initialized to the non-repeated dummy argument count
  // for other instrinsics.
  bool isMaxMin{dummyArgPatterns > 0 &&
      dummy[dummyArgPatterns - 1].optionality == Optionality::repeats};
  std::vector<ActualArgument *> actualForDummy(
      isMaxMin ? 0 : dummyArgPatterns, nullptr);
  int missingActualArguments{0};
  std::set<parser::CharBlock> maxMinKeywords;
  for (std::optional<ActualArgument> &arg : arguments) {
    if (!arg) {
      ++missingActualArguments;
    } else if (arg->isAlternateReturn()) {
      messages.Say(arg->sourceLocation(),
          "alternate return specifier not acceptable on call to intrinsic '%s'"_err_en_US,
          name);
      return std::nullopt;
    } else if (isMaxMin) {
      if (CheckMaxMinArgument(arg->keyword(), maxMinKeywords, name, messages)) {
        actualForDummy.push_back(&*arg);
      } else {
        return std::nullopt;
      }
    } else {
      bool found{false};
      int slot{missingActualArguments};
      for (std::size_t j{0}; j < dummyArgPatterns && !found; ++j) {
        if (dummy[j].optionality == Optionality::missing) {
          continue;
        }
        if (arg->keyword()) {
          found = *arg->keyword() == dummy[j].keyword;
          if (found) {
            if (const auto *previous{actualForDummy[j]}) {
              if (previous->keyword()) {
                messages.Say(*arg->keyword(),
                    "repeated keyword argument to intrinsic '%s'"_err_en_US,
                    name);
              } else {
                messages.Say(*arg->keyword(),
                    "keyword argument to intrinsic '%s' was supplied "
                    "positionally by an earlier actual argument"_err_en_US,
                    name);
              }
              return std::nullopt;
            }
          }
        } else {
          found = !actualForDummy[j] && slot-- == 0;
        }
        if (found) {
          actualForDummy[j] = &*arg;
        }
      }
      if (!found) {
        if (arg->keyword()) {
          messages.Say(*arg->keyword(),
              "unknown keyword argument to intrinsic '%s'"_err_en_US, name);
        } else {
          messages.Say(
              "too many actual arguments for intrinsic '%s'"_err_en_US, name);
        }
        return std::nullopt;
      }
    }
  }

  if (isMaxMin) {
    int nArgs{0};
    // max() / max(x) is invalid
    while ((arguments.size() + nArgs) < 2) {
      actualForDummy.push_back(nullptr);
      nArgs++;
    }

    CheckMaxMinA1A2Argument(arguments, maxMinKeywords, messages);
  }

  std::size_t dummies{actualForDummy.size()};

  // Check types and kinds of the actual arguments against the intrinsic's
  // interface.  Ensure that two or more arguments that have to have the same
  // (or compatible) type and kind do so.  Check for missing non-optional
  // arguments now, too.
  const ActualArgument *sameArg{nullptr};
  const ActualArgument *operandArg{nullptr};
  const IntrinsicDummyArgument *kindDummyArg{nullptr};
  const ActualArgument *kindArg{nullptr};
  std::optional<int> dimArg;
  for (std::size_t j{0}; j < dummies; ++j) {
    const IntrinsicDummyArgument &d{dummy[std::min(j, dummyArgPatterns - 1)]};
    if (d.typePattern.kindCode == KindCode::kindArg) {
      CHECK(!kindDummyArg);
      kindDummyArg = &d;
    }
    const ActualArgument *arg{actualForDummy[j]};
    if (!arg) {
      if (d.optionality == Optionality::required) {
        std::string kw{d.keyword};
        if (isMaxMin && maxMinKeywords.size() == 1) {
          // max(a1=x) or max(a2=x)
          const auto kwA1{dummy[0].keyword};
          const auto kwA2{dummy[1].keyword};
          if (maxMinKeywords.begin()->ToString().compare(kwA1) == 0) {
            messages.Say("missing mandatory 'a2=' argument"_err_en_US);
          } else if (maxMinKeywords.begin()->ToString().compare(kwA2) == 0) {
            messages.Say("missing mandatory 'a1=' argument"_err_en_US);
          } else {
            messages.Say(
                "missing mandatory 'a1=' and 'a2=' arguments"_err_en_US);
          }
        } else {
          messages.Say(
              "missing mandatory '%s=' argument"_err_en_US, kw.c_str());
        }
        return std::nullopt; // missing non-OPTIONAL argument
      } else {
        continue;
      }
    }
    if (d.optionality == Optionality::missing) {
      messages.Say(arg->sourceLocation(), "unexpected '%s=' argument"_err_en_US,
          d.keyword);
      return std::nullopt;
    }
    if (!d.flags.test(ArgFlag::canBeNull)) {
      // NULL() is rarely an acceptable intrinsic argument.
      if (const auto *expr{arg->UnwrapExpr()}) {
        if (IsNullPointer(*expr)) {
          messages.Say(arg->sourceLocation(),
              "A NULL() pointer is not allowed for '%s=' intrinsic argument"_err_en_US,
              d.keyword);
          return std::nullopt;
        }
      }
    }
    if (arg->GetAssumedTypeDummy()) {
      // TYPE(*) assumed-type dummy argument forwarded to intrinsic
      if (d.typePattern.categorySet == AnyType &&
          d.rank == Rank::anyOrAssumedRank &&
          (d.typePattern.kindCode == KindCode::any ||
              d.typePattern.kindCode == KindCode::addressable)) {
        continue;
      } else {
        messages.Say(arg->sourceLocation(),
            "Assumed type TYPE(*) dummy argument not allowed for '%s=' intrinsic argument"_err_en_US,
            d.keyword);
        return std::nullopt;
      }
    }
    std::optional<DynamicType> type{arg->GetType()};
    if (!type) {
      CHECK(arg->Rank() == 0);
      const Expr<SomeType> &expr{DEREF(arg->UnwrapExpr())};
      if (IsBOZLiteral(expr)) {
        if (d.typePattern.kindCode == KindCode::typeless ||
            d.rank == Rank::elementalOrBOZ) {
          continue;
        } else {
          const IntrinsicDummyArgument *nextParam{
              j + 1 < dummies ? &dummy[j + 1] : nullptr};
          if (nextParam && nextParam->rank == Rank::elementalOrBOZ) {
            messages.Say(arg->sourceLocation(),
                "Typeless (BOZ) not allowed for both '%s=' & '%s=' arguments"_err_en_US, // C7109
                d.keyword, nextParam->keyword);
          } else {
            messages.Say(arg->sourceLocation(),
                "Typeless (BOZ) not allowed for '%s=' argument"_err_en_US,
                d.keyword);
          }
        }
      } else {
        // NULL(), procedure, or procedure pointer
        CHECK(IsProcedurePointerTarget(expr));
        if (d.typePattern.kindCode == KindCode::addressable ||
            d.rank == Rank::reduceOperation) {
          continue;
        } else if (d.typePattern.kindCode == KindCode::nullPointerType) {
          continue;
        } else {
          messages.Say(arg->sourceLocation(),
              "Actual argument for '%s=' may not be a procedure"_err_en_US,
              d.keyword);
        }
      }
      return std::nullopt;
    } else if (!d.typePattern.categorySet.test(type->category())) {
      messages.Say(arg->sourceLocation(),
          "Actual argument for '%s=' has bad type '%s'"_err_en_US, d.keyword,
          type->AsFortran());
      return std::nullopt; // argument has invalid type category
    }
    bool argOk{false};
    switch (d.typePattern.kindCode) {
    case KindCode::none:
    case KindCode::typeless:
      argOk = false;
      break;
    case KindCode::teamType:
      argOk = !type->IsUnlimitedPolymorphic() &&
          type->category() == TypeCategory::Derived &&
          semantics::IsTeamType(&type->GetDerivedTypeSpec());
      break;
    case KindCode::defaultIntegerKind:
      argOk = type->kind() == defaults.GetDefaultKind(TypeCategory::Integer);
      break;
    case KindCode::defaultRealKind:
      argOk = type->kind() == defaults.GetDefaultKind(TypeCategory::Real);
      break;
    case KindCode::doublePrecision:
      argOk = type->kind() == defaults.doublePrecisionKind();
      break;
    case KindCode::defaultCharKind:
      argOk = type->kind() == defaults.GetDefaultKind(TypeCategory::Character);
      break;
    case KindCode::defaultLogicalKind:
      argOk = type->kind() == defaults.GetDefaultKind(TypeCategory::Logical);
      break;
    case KindCode::any:
      argOk = true;
      break;
    case KindCode::kindArg:
      CHECK(type->category() == TypeCategory::Integer);
      CHECK(!kindArg);
      kindArg = arg;
      argOk = true;
      break;
    case KindCode::dimArg:
      CHECK(type->category() == TypeCategory::Integer);
      dimArg = j;
      argOk = true;
      break;
    case KindCode::same:
      if (!sameArg) {
        sameArg = arg;
      }
      argOk = type->IsTkLenCompatibleWith(sameArg->GetType().value());
      break;
    case KindCode::sameKind:
      if (!sameArg) {
        sameArg = arg;
      }
      argOk = type->IsTkCompatibleWith(sameArg->GetType().value());
      break;
    case KindCode::operand:
      if (!operandArg) {
        operandArg = arg;
      } else if (auto prev{operandArg->GetType()}) {
        if (type->category() == prev->category()) {
          if (type->kind() > prev->kind()) {
            operandArg = arg;
          }
        } else if (prev->category() == TypeCategory::Integer) {
          operandArg = arg;
        }
      }
      argOk = true;
      break;
    case KindCode::effectiveKind:
      common::die("INTERNAL: KindCode::effectiveKind appears on argument '%s' "
                  "for intrinsic '%s'",
          d.keyword, name);
      break;
    case KindCode::addressable:
    case KindCode::nullPointerType:
      argOk = true;
      break;
    case KindCode::exactKind:
      argOk = type->kind() == d.typePattern.exactKindValue;
      break;
    case KindCode::sameAtom:
      if (!sameArg) {
        sameArg = arg;
        argOk = CheckAtomicKind(DEREF(arg), builtinsScope, messages);
      } else {
        argOk = type->IsTkCompatibleWith(sameArg->GetType().value());
        if (!argOk) {
          messages.Say(arg->sourceLocation(),
              "Actual argument for '%s=' must have same type and kind as 'atom=', but is '%s'"_err_en_US,
              d.keyword, type->AsFortran());
        }
      }
      if (!argOk)
        return std::nullopt;
      break;
    case KindCode::atomicIntKind:
      argOk = type->kind() ==
          GetBuiltinKind(builtinsScope, "__builtin_atomic_int_kind");
      if (!argOk) {
        messages.Say(arg->sourceLocation(),
            "Actual argument for '%s=' must have kind=atomic_int_kind, but is '%s'"_err_en_US,
            d.keyword, type->AsFortran());
        return std::nullopt;
      }
      break;
    case KindCode::atomicIntOrLogicalKind:
      argOk = CheckAtomicKind(DEREF(arg), builtinsScope, messages);
      if (!argOk)
        return std::nullopt;
      break;
    default:
      CRASH_NO_CASE;
    }
    if (!argOk) {
      messages.Say(arg->sourceLocation(),
          "Actual argument for '%s=' has bad type or kind '%s'"_err_en_US,
          d.keyword, type->AsFortran());
      return std::nullopt;
    }
  }

  // Check the ranks of the arguments against the intrinsic's interface.
  const ActualArgument *arrayArg{nullptr};
  const char *arrayArgName{nullptr};
  const ActualArgument *knownArg{nullptr};
  std::optional<int> shapeArgSize;
  int elementalRank{0};
  for (std::size_t j{0}; j < dummies; ++j) {
    const IntrinsicDummyArgument &d{dummy[std::min(j, dummyArgPatterns - 1)]};
    if (const ActualArgument *arg{actualForDummy[j]}) {
      bool isAssumedRank{IsAssumedRank(*arg)};
      if (isAssumedRank && d.rank != Rank::anyOrAssumedRank) {
        messages.Say(arg->sourceLocation(),
            "Assumed-rank array cannot be forwarded to '%s=' argument"_err_en_US,
            d.keyword);
        return std::nullopt;
      }
      int rank{arg->Rank()};
      bool argOk{false};
      switch (d.rank) {
      case Rank::elemental:
      case Rank::elementalOrBOZ:
        if (elementalRank == 0) {
          elementalRank = rank;
        }
        argOk = rank == 0 || rank == elementalRank;
        break;
      case Rank::scalar:
        argOk = rank == 0;
        break;
      case Rank::vector:
        argOk = rank == 1;
        break;
      case Rank::shape:
        CHECK(!shapeArgSize);
        if (rank != 1) {
          messages.Say(arg->sourceLocation(),
              "'shape=' argument must be an array of rank 1"_err_en_US);
          return std::nullopt;
        } else {
          if (auto shape{GetShape(context, *arg)}) {
            if (auto constShape{AsConstantShape(context, *shape)}) {
              shapeArgSize = constShape->At(ConstantSubscripts{1}).ToInt64();
              CHECK(*shapeArgSize >= 0);
              argOk = true;
            }
          }
        }
        if (!argOk) {
          messages.Say(arg->sourceLocation(),
              "'shape=' argument must be a vector of known size"_err_en_US);
          return std::nullopt;
        }
        break;
      case Rank::matrix:
        argOk = rank == 2;
        break;
      case Rank::array:
        argOk = rank > 0;
        if (!arrayArg) {
          arrayArg = arg;
          arrayArgName = d.keyword;
        }
        break;
      case Rank::coarray:
        argOk = IsCoarray(*arg);
        if (!argOk) {
          messages.Say(arg->sourceLocation(),
              "'coarray=' argument must have corank > 0 for intrinsic '%s'"_err_en_US,
              name);
          return std::nullopt;
        }
        break;
      case Rank::atom:
        argOk = rank == 0 && (IsCoarray(*arg) || ExtractCoarrayRef(*arg));
        if (!argOk) {
          messages.Say(arg->sourceLocation(),
              "'%s=' argument must be a scalar coarray or coindexed object for intrinsic '%s'"_err_en_US,
              d.keyword, name);
          return std::nullopt;
        }
        break;
      case Rank::known:
        if (!knownArg) {
          knownArg = arg;
        }
        argOk = rank == knownArg->Rank();
        break;
      case Rank::anyOrAssumedRank:
        if (!dimArg && rank > 0 && !isAssumedRank &&
            (std::strcmp(name, "shape") == 0 ||
                std::strcmp(name, "size") == 0 ||
                std::strcmp(name, "ubound") == 0)) {
          // Check for a whole assumed-size array argument.
          // These are disallowed for SHAPE, and require DIM= for
          // SIZE and UBOUND.
          // (A previous error message for UBOUND will take precedence
          // over this one, as this error is caught by the second entry
          // for UBOUND.)
          if (auto named{ExtractNamedEntity(*arg)}) {
            if (semantics::IsAssumedSizeArray(named->GetLastSymbol())) {
              if (strcmp(name, "shape") == 0) {
                messages.Say(arg->sourceLocation(),
                    "The '%s=' argument to the intrinsic function '%s' may not be assumed-size"_err_en_US,
                    d.keyword, name);
              } else {
                messages.Say(arg->sourceLocation(),
                    "A dim= argument is required for '%s' when the array is assumed-size"_err_en_US,
                    name);
              }
              return std::nullopt;
            }
          }
        }
        argOk = true;
        break;
      case Rank::conformable: // arg must be conformable with previous arrayArg
        CHECK(arrayArg);
        CHECK(arrayArgName);
        if (const std::optional<Shape> &arrayArgShape{
                GetShape(context, *arrayArg)}) {
          if (std::optional<Shape> argShape{GetShape(context, *arg)}) {
            std::string arrayArgMsg{"'"};
            arrayArgMsg = arrayArgMsg + arrayArgName + "='" + " argument";
            std::string argMsg{"'"};
            argMsg = argMsg + d.keyword + "='" + " argument";
            CheckConformance(context.messages(), *arrayArgShape, *argShape,
                CheckConformanceFlags::RightScalarExpandable,
                arrayArgMsg.c_str(), argMsg.c_str());
          }
        }
        argOk = true; // Avoid an additional error message
        break;
      case Rank::dimReduced:
      case Rank::dimRemovedOrScalar:
        CHECK(arrayArg);
        argOk = rank == 0 || rank + 1 == arrayArg->Rank();
        break;
      case Rank::reduceOperation:
        // The reduction function is validated in ApplySpecificChecks().
        argOk = true;
        break;
      case Rank::scalarIfDim:
      case Rank::locReduced:
      case Rank::rankPlus1:
      case Rank::shaped:
        common::die("INTERNAL: result-only rank code appears on argument '%s' "
                    "for intrinsic '%s'",
            d.keyword, name);
      }
      if (!argOk) {
        messages.Say(arg->sourceLocation(),
            "'%s=' argument has unacceptable rank %d"_err_en_US, d.keyword,
            rank);
        return std::nullopt;
      }
    }
  }

  // Calculate the characteristics of the function result, if any
  std::optional<DynamicType> resultType;
  if (auto category{result.categorySet.LeastElement()}) {
    // The intrinsic is not a subroutine.
    if (call.isSubroutineCall) {
      return std::nullopt;
    }
    switch (result.kindCode) {
    case KindCode::defaultIntegerKind:
      CHECK(result.categorySet == IntType);
      CHECK(*category == TypeCategory::Integer);
      resultType = DynamicType{TypeCategory::Integer,
          defaults.GetDefaultKind(TypeCategory::Integer)};
      break;
    case KindCode::defaultRealKind:
      CHECK(result.categorySet == CategorySet{*category});
      CHECK(FloatingType.test(*category));
      resultType =
          DynamicType{*category, defaults.GetDefaultKind(TypeCategory::Real)};
      break;
    case KindCode::doublePrecision:
      CHECK(result.categorySet == CategorySet{*category});
      CHECK(FloatingType.test(*category));
      resultType = DynamicType{*category, defaults.doublePrecisionKind()};
      break;
    case KindCode::defaultLogicalKind:
      CHECK(result.categorySet == LogicalType);
      CHECK(*category == TypeCategory::Logical);
      resultType = DynamicType{TypeCategory::Logical,
          defaults.GetDefaultKind(TypeCategory::Logical)};
      break;
    case KindCode::same:
      CHECK(sameArg);
      if (std::optional<DynamicType> aType{sameArg->GetType()}) {
        if (result.categorySet.test(aType->category())) {
          resultType = *aType;
        } else {
          resultType = DynamicType{*category, aType->kind()};
        }
      }
      break;
    case KindCode::sameKind:
      CHECK(sameArg);
      if (std::optional<DynamicType> aType{sameArg->GetType()}) {
        resultType = DynamicType{*category, aType->kind()};
      }
      break;
    case KindCode::operand:
      CHECK(operandArg);
      resultType = operandArg->GetType();
      CHECK(!resultType || result.categorySet.test(resultType->category()));
      break;
    case KindCode::effectiveKind:
      CHECK(kindDummyArg);
      CHECK(result.categorySet == CategorySet{*category});
      if (kindArg) {
        if (auto *expr{kindArg->UnwrapExpr()}) {
          CHECK(expr->Rank() == 0);
          if (auto code{ToInt64(*expr)}) {
            if (context.targetCharacteristics().IsTypeEnabled(
                    *category, *code)) {
              if (*category == TypeCategory::Character) { // ACHAR & CHAR
                resultType = DynamicType{static_cast<int>(*code), 1};
              } else {
                resultType = DynamicType{*category, static_cast<int>(*code)};
              }
              break;
            }
          }
        }
        messages.Say("'kind=' argument must be a constant scalar integer "
                     "whose value is a supported kind for the "
                     "intrinsic result type"_err_en_US);
        return std::nullopt;
      } else if (kindDummyArg->flags.test(ArgFlag::defaultsToSameKind)) {
        CHECK(sameArg);
        resultType = *sameArg->GetType();
      } else if (kindDummyArg->flags.test(ArgFlag::defaultsToSizeKind)) {
        CHECK(*category == TypeCategory::Integer);
        resultType =
            DynamicType{TypeCategory::Integer, defaults.sizeIntegerKind()};
      } else {
        CHECK(kindDummyArg->flags.test(ArgFlag::defaultsToDefaultForResult));
        int kind{defaults.GetDefaultKind(*category)};
        if (*category == TypeCategory::Character) { // ACHAR & CHAR
          resultType = DynamicType{kind, 1};
        } else {
          resultType = DynamicType{*category, kind};
        }
      }
      break;
    case KindCode::likeMultiply:
      CHECK(dummies >= 2);
      CHECK(actualForDummy[0]);
      CHECK(actualForDummy[1]);
      resultType = actualForDummy[0]->GetType()->ResultTypeForMultiply(
          *actualForDummy[1]->GetType());
      break;
    case KindCode::subscript:
      CHECK(result.categorySet == IntType);
      CHECK(*category == TypeCategory::Integer);
      resultType =
          DynamicType{TypeCategory::Integer, defaults.subscriptIntegerKind()};
      break;
    case KindCode::size:
      CHECK(result.categorySet == IntType);
      CHECK(*category == TypeCategory::Integer);
      resultType =
          DynamicType{TypeCategory::Integer, defaults.sizeIntegerKind()};
      break;
    case KindCode::teamType:
      CHECK(result.categorySet == DerivedType);
      CHECK(*category == TypeCategory::Derived);
      resultType = DynamicType{
          GetBuiltinDerivedType(builtinsScope, "__builtin_team_type")};
      break;
    case KindCode::exactKind:
      resultType = DynamicType{*category, result.exactKindValue};
      break;
    case KindCode::defaultCharKind:
    case KindCode::typeless:
    case KindCode::any:
    case KindCode::kindArg:
    case KindCode::dimArg:
      common::die(
          "INTERNAL: bad KindCode appears on intrinsic '%s' result", name);
      break;
    default:
      CRASH_NO_CASE;
    }
  } else {
    if (!call.isSubroutineCall) {
      return std::nullopt;
    }
    CHECK(result.kindCode == KindCode::none);
  }

  // Emit warnings when the syntactic presence of a DIM= argument determines
  // the semantics of the call but the associated actual argument may not be
  // present at execution time.
  if (dimArg) {
    std::optional<int> arrayRank;
    if (arrayArg) {
      arrayRank = arrayArg->Rank();
      if (auto dimVal{ToInt64(actualForDummy[*dimArg])}) {
        if (*dimVal < 1) {
          messages.Say(
              "The value of DIM= (%jd) may not be less than 1"_err_en_US,
              static_cast<std::intmax_t>(*dimVal));
        } else if (*dimVal > *arrayRank) {
          messages.Say(
              "The value of DIM= (%jd) may not be greater than %d"_err_en_US,
              static_cast<std::intmax_t>(*dimVal), *arrayRank);
        }
      }
    }
    switch (rank) {
    case Rank::dimReduced:
    case Rank::dimRemovedOrScalar:
    case Rank::locReduced:
    case Rank::scalarIfDim:
      if (dummy[*dimArg].optionality == Optionality::required) {
        if (const Symbol *whole{
                UnwrapWholeSymbolOrComponentDataRef(actualForDummy[*dimArg])}) {
          if (IsOptional(*whole) || IsAllocatableOrPointer(*whole)) {
            if (rank == Rank::scalarIfDim || arrayRank.value_or(-1) == 1) {
              messages.Say(
                  "The actual argument for DIM= is optional, pointer, or allocatable, and it is assumed to be present and equal to 1 at execution time"_port_en_US);
            } else {
              messages.Say(
                  "The actual argument for DIM= is optional, pointer, or allocatable, and may not be absent during execution; parenthesize to silence this warning"_warn_en_US);
            }
          }
        }
      }
      break;
    default:;
    }
  }

  // At this point, the call is acceptable.
  // Determine the rank of the function result.
  int resultRank{0};
  switch (rank) {
  case Rank::elemental:
    resultRank = elementalRank;
    break;
  case Rank::scalar:
    resultRank = 0;
    break;
  case Rank::vector:
    resultRank = 1;
    break;
  case Rank::matrix:
    resultRank = 2;
    break;
  case Rank::conformable:
    CHECK(arrayArg);
    resultRank = arrayArg->Rank();
    break;
  case Rank::dimReduced:
    CHECK(arrayArg);
    resultRank = dimArg ? arrayArg->Rank() - 1 : 0;
    break;
  case Rank::locReduced:
    CHECK(arrayArg);
    resultRank = dimArg ? arrayArg->Rank() - 1 : 1;
    break;
  case Rank::rankPlus1:
    CHECK(knownArg);
    resultRank = knownArg->Rank() + 1;
    break;
  case Rank::shaped:
    CHECK(shapeArgSize);
    resultRank = *shapeArgSize;
    break;
  case Rank::scalarIfDim:
    resultRank = dimArg ? 0 : 1;
    break;
  case Rank::elementalOrBOZ:
  case Rank::shape:
  case Rank::array:
  case Rank::coarray:
  case Rank::atom:
  case Rank::known:
  case Rank::anyOrAssumedRank:
  case Rank::reduceOperation:
  case Rank::dimRemovedOrScalar:
    common::die("INTERNAL: bad Rank code on intrinsic '%s' result", name);
    break;
  }
  CHECK(resultRank >= 0);

  // Rearrange the actual arguments into dummy argument order.
  ActualArguments rearranged(dummies);
  for (std::size_t j{0}; j < dummies; ++j) {
    if (ActualArgument *arg{actualForDummy[j]}) {
      rearranged[j] = std::move(*arg);
    }
  }

  // Characterize the specific intrinsic procedure.
  characteristics::DummyArguments dummyArgs;
  std::optional<int> sameDummyArg;

  for (std::size_t j{0}; j < dummies; ++j) {
    const IntrinsicDummyArgument &d{dummy[std::min(j, dummyArgPatterns - 1)]};
    if (const auto &arg{rearranged[j]}) {
      if (const Expr<SomeType> *expr{arg->UnwrapExpr()}) {
        std::string kw{d.keyword};
        if (arg->keyword()) {
          kw = arg->keyword()->ToString();
        } else if (isMaxMin) {
          for (std::size_t k{j + 1};; ++k) {
            kw = "a"s + std::to_string(k);
            auto iter{std::find_if(dummyArgs.begin(), dummyArgs.end(),
                [&kw](const characteristics::DummyArgument &prev) {
                  return prev.name == kw;
                })};
            if (iter == dummyArgs.end()) {
              break;
            }
          }
        }
        auto dc{characteristics::DummyArgument::FromActual(
            std::move(kw), *expr, context)};
        if (!dc) {
          common::die("INTERNAL: could not characterize intrinsic function "
                      "actual argument '%s'",
              expr->AsFortran().c_str());
          return std::nullopt;
        }
        dummyArgs.emplace_back(std::move(*dc));
        if (d.typePattern.kindCode == KindCode::same && !sameDummyArg) {
          sameDummyArg = j;
        }
      } else {
        CHECK(arg->GetAssumedTypeDummy());
        dummyArgs.emplace_back(std::string{d.keyword},
            characteristics::DummyDataObject{DynamicType::AssumedType()});
      }
    } else {
      // optional argument is absent
      CHECK(d.optionality != Optionality::required);
      if (d.typePattern.kindCode == KindCode::same) {
        dummyArgs.emplace_back(dummyArgs[sameDummyArg.value()]);
      } else {
        auto category{d.typePattern.categorySet.LeastElement().value()};
        if (category == TypeCategory::Derived) {
          // TODO: any other built-in derived types used as optional intrinsic
          // dummies?
          CHECK(d.typePattern.kindCode == KindCode::teamType);
          characteristics::TypeAndShape typeAndShape{
              GetBuiltinDerivedType(builtinsScope, "__builtin_team_type")};
          dummyArgs.emplace_back(std::string{d.keyword},
              characteristics::DummyDataObject{std::move(typeAndShape)});
        } else {
          characteristics::TypeAndShape typeAndShape{
              DynamicType{category, defaults.GetDefaultKind(category)}};
          dummyArgs.emplace_back(std::string{d.keyword},
              characteristics::DummyDataObject{std::move(typeAndShape)});
        }
      }
      dummyArgs.back().SetOptional();
    }
    dummyArgs.back().SetIntent(d.intent);
  }
  characteristics::Procedure::Attrs attrs;
  if (elementalRank > 0) {
    attrs.set(characteristics::Procedure::Attr::Elemental);
  }
  if (call.isSubroutineCall) {
    return SpecificCall{
        SpecificIntrinsic{
            name, characteristics::Procedure{std::move(dummyArgs), attrs}},
        std::move(rearranged)};
  } else {
    attrs.set(characteristics::Procedure::Attr::Pure);
    characteristics::TypeAndShape typeAndShape{resultType.value(), resultRank};
    characteristics::FunctionResult funcResult{std::move(typeAndShape)};
    characteristics::Procedure chars{
        std::move(funcResult), std::move(dummyArgs), attrs};
    return SpecificCall{
        SpecificIntrinsic{name, std::move(chars)}, std::move(rearranged)};
  }
}

class IntrinsicProcTable::Implementation {
public:
  explicit Implementation(const common::IntrinsicTypeDefaultKinds &dfts)
      : defaults_{dfts} {
    for (const IntrinsicInterface &f : genericIntrinsicFunction) {
      genericFuncs_.insert(std::make_pair(std::string{f.name}, &f));
    }
    for (const std::pair<const char *, const char *> &a : genericAlias) {
      aliases_.insert(
          std::make_pair(std::string{a.first}, std::string{a.second}));
    }
    for (const SpecificIntrinsicInterface &f : specificIntrinsicFunction) {
      specificFuncs_.insert(std::make_pair(std::string{f.name}, &f));
    }
    for (const IntrinsicInterface &f : intrinsicSubroutine) {
      subroutines_.insert(std::make_pair(std::string{f.name}, &f));
    }
  }

  void SupplyBuiltins(const semantics::Scope &builtins) {
    builtinsScope_ = &builtins;
  }

  bool IsIntrinsic(const std::string &) const;
  bool IsIntrinsicFunction(const std::string &) const;
  bool IsIntrinsicSubroutine(const std::string &) const;

  IntrinsicClass GetIntrinsicClass(const std::string &) const;
  std::string GetGenericIntrinsicName(const std::string &) const;

  std::optional<SpecificCall> Probe(
      const CallCharacteristics &, ActualArguments &, FoldingContext &) const;

  std::optional<SpecificIntrinsicFunctionInterface> IsSpecificIntrinsicFunction(
      const std::string &) const;

  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;

private:
  DynamicType GetSpecificType(const TypePattern &) const;
  SpecificCall HandleNull(ActualArguments &, FoldingContext &) const;
  std::optional<SpecificCall> HandleC_F_Pointer(
      ActualArguments &, FoldingContext &) const;
  const std::string &ResolveAlias(const std::string &name) const {
    auto iter{aliases_.find(name)};
    return iter == aliases_.end() ? name : iter->second;
  }

  common::IntrinsicTypeDefaultKinds defaults_;
  std::multimap<std::string, const IntrinsicInterface *> genericFuncs_;
  std::multimap<std::string, const SpecificIntrinsicInterface *> specificFuncs_;
  std::multimap<std::string, const IntrinsicInterface *> subroutines_;
  const semantics::Scope *builtinsScope_{nullptr};
  std::map<std::string, std::string> aliases_;
};

bool IntrinsicProcTable::Implementation::IsIntrinsicFunction(
    const std::string &name0) const {
  const std::string &name{ResolveAlias(name0)};
  auto specificRange{specificFuncs_.equal_range(name)};
  if (specificRange.first != specificRange.second) {
    return true;
  }
  auto genericRange{genericFuncs_.equal_range(name)};
  if (genericRange.first != genericRange.second) {
    return true;
  }
  // special cases
  return name == "null";
}
bool IntrinsicProcTable::Implementation::IsIntrinsicSubroutine(
    const std::string &name) const {
  auto subrRange{subroutines_.equal_range(name)};
  if (subrRange.first != subrRange.second) {
    return true;
  }
  // special cases
  return name == "__builtin_c_f_pointer";
}
bool IntrinsicProcTable::Implementation::IsIntrinsic(
    const std::string &name) const {
  return IsIntrinsicFunction(name) || IsIntrinsicSubroutine(name);
}

IntrinsicClass IntrinsicProcTable::Implementation::GetIntrinsicClass(
    const std::string &name) const {
  auto specificIntrinsic{specificFuncs_.find(name)};
  if (specificIntrinsic != specificFuncs_.end()) {
    return specificIntrinsic->second->intrinsicClass;
  }
  auto genericIntrinsic{genericFuncs_.find(name)};
  if (genericIntrinsic != genericFuncs_.end()) {
    return genericIntrinsic->second->intrinsicClass;
  }
  auto subrIntrinsic{subroutines_.find(name)};
  if (subrIntrinsic != subroutines_.end()) {
    return subrIntrinsic->second->intrinsicClass;
  }
  return IntrinsicClass::noClass;
}

std::string IntrinsicProcTable::Implementation::GetGenericIntrinsicName(
    const std::string &name) const {
  auto specificIntrinsic{specificFuncs_.find(name)};
  if (specificIntrinsic != specificFuncs_.end()) {
    if (const char *genericName{specificIntrinsic->second->generic}) {
      return {genericName};
    }
  }
  return name;
}

bool CheckAndRearrangeArguments(ActualArguments &arguments,
    parser::ContextualMessages &messages, const char *const dummyKeywords[],
    std::size_t trailingOptionals) {
  std::size_t numDummies{0};
  while (dummyKeywords[numDummies]) {
    ++numDummies;
  }
  CHECK(trailingOptionals <= numDummies);
  if (arguments.size() > numDummies) {
    messages.Say("Too many actual arguments (%zd > %zd)"_err_en_US,
        arguments.size(), numDummies);
    return false;
  }
  ActualArguments rearranged(numDummies);
  bool anyKeywords{false};
  std::size_t position{0};
  for (std::optional<ActualArgument> &arg : arguments) {
    std::size_t dummyIndex{0};
    if (arg && arg->keyword()) {
      anyKeywords = true;
      for (; dummyIndex < numDummies; ++dummyIndex) {
        if (*arg->keyword() == dummyKeywords[dummyIndex]) {
          break;
        }
      }
      if (dummyIndex >= numDummies) {
        messages.Say(*arg->keyword(),
            "Unknown argument keyword '%s='"_err_en_US, *arg->keyword());
        return false;
      }
    } else if (anyKeywords) {
      messages.Say(arg ? arg->sourceLocation() : messages.at(),
          "A positional actual argument may not appear after any keyword arguments"_err_en_US);
      return false;
    } else {
      dummyIndex = position++;
    }
    if (rearranged[dummyIndex]) {
      messages.Say(arg ? arg->sourceLocation() : messages.at(),
          "Dummy argument '%s=' appears more than once"_err_en_US,
          dummyKeywords[dummyIndex]);
      return false;
    }
    rearranged[dummyIndex] = std::move(arg);
    arg.reset();
  }
  bool anyMissing{false};
  for (std::size_t j{0}; j < numDummies - trailingOptionals; ++j) {
    if (!rearranged[j]) {
      messages.Say("Dummy argument '%s=' is absent and not OPTIONAL"_err_en_US,
          dummyKeywords[j]);
      anyMissing = true;
    }
  }
  arguments = std::move(rearranged);
  return !anyMissing;
}

// The NULL() intrinsic is a special case.
SpecificCall IntrinsicProcTable::Implementation::HandleNull(
    ActualArguments &arguments, FoldingContext &context) const {
  static const char *const keywords[]{"mold", nullptr};
  if (CheckAndRearrangeArguments(arguments, context.messages(), keywords, 1) &&
      arguments[0]) {
    if (Expr<SomeType> * mold{arguments[0]->UnwrapExpr()}) {
      bool isProcPtrTarget{IsProcedurePointerTarget(*mold)};
      if (isProcPtrTarget || IsAllocatableOrPointerObject(*mold, context)) {
        characteristics::DummyArguments args;
        std::optional<characteristics::FunctionResult> fResult;
        if (isProcPtrTarget) {
          // MOLD= procedure pointer
          const Symbol *last{GetLastSymbol(*mold)};
          CHECK(last);
          auto procPointer{IsProcedure(*last)
                  ? characteristics::Procedure::Characterize(*last, context)
                  : std::nullopt};
          // procPointer is null if there was an error with the analysis
          // associated with the procedure pointer
          if (procPointer) {
            args.emplace_back("mold"s,
                characteristics::DummyProcedure{common::Clone(*procPointer)});
            fResult.emplace(std::move(*procPointer));
          }
        } else if (auto type{mold->GetType()}) {
          // MOLD= object pointer
          characteristics::TypeAndShape typeAndShape{
              *type, GetShape(context, *mold)};
          args.emplace_back(
              "mold"s, characteristics::DummyDataObject{typeAndShape});
          fResult.emplace(std::move(typeAndShape));
        } else {
          context.messages().Say(arguments[0]->sourceLocation(),
              "MOLD= argument to NULL() lacks type"_err_en_US);
        }
        if (fResult) {
          fResult->attrs.set(characteristics::FunctionResult::Attr::Pointer);
          characteristics::Procedure::Attrs attrs;
          attrs.set(characteristics::Procedure::Attr::NullPointer);
          characteristics::Procedure chars{
              std::move(*fResult), std::move(args), attrs};
          return SpecificCall{SpecificIntrinsic{"null"s, std::move(chars)},
              std::move(arguments)};
        }
      }
    }
    context.messages().Say(arguments[0]->sourceLocation(),
        "MOLD= argument to NULL() must be a pointer or allocatable"_err_en_US);
  }
  characteristics::Procedure::Attrs attrs;
  attrs.set(characteristics::Procedure::Attr::NullPointer);
  attrs.set(characteristics::Procedure::Attr::Pure);
  arguments.clear();
  return SpecificCall{
      SpecificIntrinsic{"null"s,
          characteristics::Procedure{characteristics::DummyArguments{}, attrs}},
      std::move(arguments)};
}

// Subroutine C_F_POINTER(CPTR=,FPTR=[,SHAPE=]) from
// intrinsic module ISO_C_BINDING (18.2.3.3)
std::optional<SpecificCall>
IntrinsicProcTable::Implementation::HandleC_F_Pointer(
    ActualArguments &arguments, FoldingContext &context) const {
  characteristics::Procedure::Attrs attrs;
  attrs.set(characteristics::Procedure::Attr::Subroutine);
  static const char *const keywords[]{"cptr", "fptr", "shape", nullptr};
  characteristics::DummyArguments dummies;
  if (CheckAndRearrangeArguments(arguments, context.messages(), keywords, 1)) {
    CHECK(arguments.size() == 3);
    if (const auto *expr{arguments[0].value().UnwrapExpr()}) {
      // General semantic checks will catch an actual argument that's not
      // scalar.
      if (auto type{expr->GetType()}) {
        if (type->category() != TypeCategory::Derived ||
            type->IsPolymorphic() ||
            type->GetDerivedTypeSpec().typeSymbol().name() !=
                "__builtin_c_ptr") {
          context.messages().Say(arguments[0]->sourceLocation(),
              "CPTR= argument to C_F_POINTER() must be a C_PTR"_err_en_US);
        }
        characteristics::DummyDataObject cptr{
            characteristics::TypeAndShape{*type}};
        cptr.intent = common::Intent::In;
        dummies.emplace_back("cptr"s, std::move(cptr));
      }
    }
    if (const auto *expr{arguments[1].value().UnwrapExpr()}) {
      int fptrRank{expr->Rank()};
      if (auto type{expr->GetType()}) {
        if (type->HasDeferredTypeParameter()) {
          context.messages().Say(arguments[1]->sourceLocation(),
              "FPTR= argument to C_F_POINTER() may not have a deferred type parameter"_err_en_US);
        }
        if (ExtractCoarrayRef(*expr)) {
          context.messages().Say(arguments[1]->sourceLocation(),
              "FPTR= argument to C_F_POINTER() may not be a coindexed object"_err_en_US);
        }
        characteristics::DummyDataObject fptr{
            characteristics::TypeAndShape{*type, fptrRank}};
        fptr.intent = common::Intent::Out;
        fptr.attrs.set(characteristics::DummyDataObject::Attr::Pointer);
        dummies.emplace_back("fptr"s, std::move(fptr));
      } else {
        context.messages().Say(arguments[1]->sourceLocation(),
            "FPTR= argument to C_F_POINTER() must have a type"_err_en_US);
      }
      if (arguments[2] && fptrRank == 0) {
        context.messages().Say(arguments[2]->sourceLocation(),
            "SHAPE= argument to C_F_POINTER() may not appear when FPTR= is scalar"_err_en_US);
      } else if (!arguments[2] && fptrRank > 0) {
        context.messages().Say(
            "SHAPE= argument to C_F_POINTER() must appear when FPTR= is an array"_err_en_US);
      } else if (arguments[2]) {
        if (const auto *argExpr{arguments[2].value().UnwrapExpr()}) {
          if (argExpr->Rank() > 1) {
            context.messages().Say(arguments[2]->sourceLocation(),
                "SHAPE= argument to C_F_POINTER() must be a rank-one array."_err_en_US);
          } else if (argExpr->Rank() == 1) {
            if (auto constShape{GetConstantShape(context, *argExpr)}) {
              if (constShape->At(ConstantSubscripts{1}).ToInt64() != fptrRank) {
                context.messages().Say(arguments[2]->sourceLocation(),
                    "SHAPE= argument to C_F_POINTER() must have size equal to the rank of FPTR="_err_en_US);
              }
            }
          }
        }
      }
    }
  }
  if (dummies.size() == 2) {
    DynamicType shapeType{TypeCategory::Integer, defaults_.sizeIntegerKind()};
    if (arguments[2]) {
      if (auto type{arguments[2]->GetType()}) {
        if (type->category() == TypeCategory::Integer) {
          shapeType = *type;
        }
      }
    }
    characteristics::DummyDataObject shape{
        characteristics::TypeAndShape{shapeType, 1}};
    shape.intent = common::Intent::In;
    shape.attrs.set(characteristics::DummyDataObject::Attr::Optional);
    dummies.emplace_back("shape"s, std::move(shape));
    return SpecificCall{
        SpecificIntrinsic{"__builtin_c_f_pointer"s,
            characteristics::Procedure{std::move(dummies), attrs}},
        std::move(arguments)};
  } else {
    return std::nullopt;
  }
}

static bool CheckAssociated(SpecificCall &call, FoldingContext &context) {
  bool ok{true};
  if (const auto &pointerArg{call.arguments[0]}) {
    if (const auto *pointerExpr{pointerArg->UnwrapExpr()}) {
      if (const Symbol *pointerSymbol{GetLastSymbol(*pointerExpr)}) {
        if (!pointerSymbol->attrs().test(semantics::Attr::POINTER)) {
          AttachDeclaration(context.messages().Say(pointerArg->sourceLocation(),
                                "POINTER= argument of ASSOCIATED() must be a "
                                "POINTER"_err_en_US),
              *pointerSymbol);
        } else {
          if (const auto &targetArg{call.arguments[1]}) {
            if (const auto *targetExpr{targetArg->UnwrapExpr()}) {
              std::optional<characteristics::Procedure> pointerProc, targetProc;
              const auto *targetProcDesignator{
                  UnwrapExpr<ProcedureDesignator>(*targetExpr)};
              const Symbol *targetSymbol{GetLastSymbol(*targetExpr)};
              bool isCall{false};
              std::string targetName;
              if (const auto *targetProcRef{// target is a function call
                      std::get_if<ProcedureRef>(&targetExpr->u)}) {
                if (auto targetRefedChars{
                        characteristics::Procedure::Characterize(
                            *targetProcRef, context)}) {
                  targetProc = *targetRefedChars;
                  targetName = targetProcRef->proc().GetName() + "()";
                  isCall = true;
                }
              } else if (targetProcDesignator) {
                targetProc = characteristics::Procedure::Characterize(
                    *targetProcDesignator, context);
                targetName = targetProcDesignator->GetName();
              } else if (targetSymbol) {
                if (IsProcedure(*targetSymbol)) {
                  // proc that's not a call
                  targetProc = characteristics::Procedure::Characterize(
                      *targetSymbol, context);
                }
                targetName = targetSymbol->name().ToString();
              }
              if (IsProcedure(*pointerSymbol)) {
                pointerProc = characteristics::Procedure::Characterize(
                    *pointerSymbol, context);
              }
              if (pointerProc) {
                if (targetProc) {
                  // procedure pointer and procedure target
                  std::string whyNot;
                  const SpecificIntrinsic *specificIntrinsic{nullptr};
                  if (targetProcDesignator) {
                    specificIntrinsic =
                        targetProcDesignator->GetSpecificIntrinsic();
                  }
                  if (std::optional<parser::MessageFixedText> msg{
                          CheckProcCompatibility(isCall, pointerProc,
                              &*targetProc, specificIntrinsic, whyNot)}) {
                    msg->set_severity(parser::Severity::Warning);
                    AttachDeclaration(
                        context.messages().Say(std::move(*msg),
                            "pointer '" + pointerSymbol->name().ToString() +
                                "'",
                            targetName, whyNot),
                        *pointerSymbol);
                  }
                } else if (!IsNullProcedurePointer(*targetExpr)) {
                  // procedure pointer and object target
                  AttachDeclaration(
                      context.messages().Say(
                          "POINTER= argument '%s' is a procedure "
                          "pointer but the TARGET= argument '%s' is not a "
                          "procedure or procedure pointer"_err_en_US,
                          pointerSymbol->name(), targetName),
                      *pointerSymbol);
                }
              } else if (targetProc) {
                // object pointer and procedure target
                AttachDeclaration(
                    context.messages().Say(
                        "POINTER= argument '%s' is an object pointer "
                        "but the TARGET= argument '%s' is a "
                        "procedure designator"_err_en_US,
                        pointerSymbol->name(), targetName),
                    *pointerSymbol);
              } else if (targetSymbol) {
                // object pointer and target
                SymbolVector symbols{GetSymbolVector(*targetExpr)};
                CHECK(!symbols.empty());
                if (!GetLastTarget(symbols)) {
                  parser::Message *msg{context.messages().Say(
                      targetArg->sourceLocation(),
                      "TARGET= argument '%s' must have either the POINTER or the TARGET attribute"_err_en_US,
                      targetExpr->AsFortran())};
                  for (SymbolRef ref : symbols) {
                    msg = AttachDeclaration(msg, *ref);
                  }
                } else if (HasVectorSubscript(*targetExpr) ||
                    ExtractCoarrayRef(*targetExpr)) {
                  context.messages().Say(targetArg->sourceLocation(),
                      "TARGET= argument '%s' may not have a vector subscript or coindexing"_err_en_US,
                      targetExpr->AsFortran());
                }
                if (const auto pointerType{pointerArg->GetType()}) {
                  if (const auto targetType{targetArg->GetType()}) {
                    ok = pointerType->IsTkCompatibleWith(*targetType);
                  }
                }
              }
            }
          }
        }
      }
    }
  } else {
    // No arguments to ASSOCIATED()
    ok = false;
  }
  if (!ok) {
    context.messages().Say(
        "Arguments of ASSOCIATED() must be a POINTER and an optional valid target"_err_en_US);
  }
  return ok;
}

static bool CheckForNonPositiveValues(FoldingContext &context,
    const ActualArgument &arg, const std::string &procName,
    const std::string &argName) {
  bool ok{true};
  if (arg.Rank() > 0) {
    if (const Expr<SomeType> *expr{arg.UnwrapExpr()}) {
      if (const auto *intExpr{std::get_if<Expr<SomeInteger>>(&expr->u)}) {
        std::visit(
            [&](const auto &kindExpr) {
              using IntType = typename std::decay_t<decltype(kindExpr)>::Result;
              if (const auto *constArray{
                      UnwrapConstantValue<IntType>(kindExpr)}) {
                for (std::size_t j{0}; j < constArray->size(); ++j) {
                  auto arrayExpr{constArray->values().at(j)};
                  if (arrayExpr.IsNegative() || arrayExpr.IsZero()) {
                    ok = false;
                    context.messages().Say(arg.sourceLocation(),
                        "'%s=' argument for intrinsic '%s' must contain all positive values"_err_en_US,
                        argName, procName);
                  }
                }
              }
            },
            intExpr->u);
      }
    }
  } else {
    if (auto val{ToInt64(arg.UnwrapExpr())}) {
      if (*val <= 0) {
        ok = false;
        context.messages().Say(arg.sourceLocation(),
            "'%s=' argument for intrinsic '%s' must be a positive value, but is %jd"_err_en_US,
            argName, procName, static_cast<std::intmax_t>(*val));
      }
    }
  }
  return ok;
}

static bool CheckDimAgainstCorank(SpecificCall &call, FoldingContext &context) {
  bool ok{true};
  if (const auto &coarrayArg{call.arguments[0]}) {
    if (const auto &dimArg{call.arguments[1]}) {
      if (const auto *symbol{
              UnwrapWholeSymbolDataRef(coarrayArg->UnwrapExpr())}) {
        const auto corank = symbol->Corank();
        if (const auto dimNum{ToInt64(dimArg->UnwrapExpr())}) {
          if (dimNum < 1 || dimNum > corank) {
            ok = false;
            context.messages().Say(dimArg->sourceLocation(),
                "DIM=%jd dimension is out of range for coarray with corank %d"_err_en_US,
                static_cast<std::intmax_t>(*dimNum), corank);
          }
        }
      }
    }
  }
  return ok;
}

static bool CheckForCoindexedObject(FoldingContext &context,
    const std::optional<ActualArgument> &arg, const std::string &procName,
    const std::string &argName) {
  bool ok{true};
  if (arg) {
    if (ExtractCoarrayRef(arg->UnwrapExpr())) {
      ok = false;
      context.messages().Say(arg->sourceLocation(),
          "'%s' argument to '%s' may not be a coindexed object"_err_en_US,
          argName, procName);
    }
  }
  return ok;
}

static bool CheckAtomicDefineAndRef(FoldingContext &context,
    const std::optional<ActualArgument> &atomArg,
    const std::optional<ActualArgument> &valueArg,
    const std::optional<ActualArgument> &statArg, const std::string &procName) {
  bool sameType{true};
  if (valueArg && atomArg) {
    // for atomic_define and atomic_ref, 'value' arg must be the same type as
    // 'atom', but it doesn't have to be the same kind
    if (valueArg->GetType()->category() != atomArg->GetType()->category()) {
      sameType = false;
      context.messages().Say(valueArg->sourceLocation(),
          "'value=' argument to '%s' must have same type as 'atom=', but is '%s'"_err_en_US,
          procName, valueArg->GetType()->AsFortran());
    }
  }

  return sameType &&
      CheckForCoindexedObject(context, statArg, procName, "stat");
}

// Applies any semantic checks peculiar to an intrinsic.
static bool ApplySpecificChecks(SpecificCall &call, FoldingContext &context) {
  bool ok{true};
  const std::string &name{call.specificIntrinsic.name};
  if (name == "allocated") {
    const auto &arg{call.arguments[0]};
    if (arg) {
      if (const auto *expr{arg->UnwrapExpr()}) {
        ok = evaluate::IsAllocatableDesignator(*expr);
      }
    }
    if (!ok) {
      context.messages().Say(
          arg ? arg->sourceLocation() : context.messages().at(),
          "Argument of ALLOCATED() must be an ALLOCATABLE object or component"_err_en_US);
    }
  } else if (name == "associated") {
    return CheckAssociated(call, context);
  } else if (name == "atomic_and" || name == "atomic_or" ||
      name == "atomic_xor") {
    return CheckForCoindexedObject(context, call.arguments[2], name, "stat");
  } else if (name == "atomic_cas") {
    return CheckForCoindexedObject(context, call.arguments[4], name, "stat");
  } else if (name == "atomic_define") {
    return CheckAtomicDefineAndRef(
        context, call.arguments[0], call.arguments[1], call.arguments[2], name);
  } else if (name == "atomic_fetch_add" || name == "atomic_fetch_and" ||
      name == "atomic_fetch_or" || name == "atomic_fetch_xor") {
    return CheckForCoindexedObject(context, call.arguments[3], name, "stat");
  } else if (name == "atomic_ref") {
    return CheckAtomicDefineAndRef(
        context, call.arguments[1], call.arguments[0], call.arguments[2], name);
  } else if (name == "co_broadcast" || name == "co_max" || name == "co_min" ||
      name == "co_sum") {
    bool aOk{CheckForCoindexedObject(context, call.arguments[0], name, "a")};
    bool statOk{
        CheckForCoindexedObject(context, call.arguments[2], name, "stat")};
    bool errmsgOk{
        CheckForCoindexedObject(context, call.arguments[3], name, "errmsg")};
    ok = aOk && statOk && errmsgOk;
  } else if (name == "image_status") {
    if (const auto &arg{call.arguments[0]}) {
      ok = CheckForNonPositiveValues(context, *arg, name, "image");
    }
  } else if (name == "lcobound") {
    return CheckDimAgainstCorank(call, context);
  } else if (name == "loc") {
    const auto &arg{call.arguments[0]};
    ok =
        arg && (arg->GetAssumedTypeDummy() || GetLastSymbol(arg->UnwrapExpr()));
    if (!ok) {
      context.messages().Say(
          arg ? arg->sourceLocation() : context.messages().at(),
          "Argument of LOC() must be an object or procedure"_err_en_US);
    }
  } else if (name == "move_alloc") {
    ok &= CheckForCoindexedObject(context, call.arguments[0], name, "from");
    ok &= CheckForCoindexedObject(context, call.arguments[1], name, "to");
    ok &= CheckForCoindexedObject(context, call.arguments[2], name, "stat");
    ok &= CheckForCoindexedObject(context, call.arguments[3], name, "errmsg");
    if (call.arguments[0] && call.arguments[1]) {
      for (int j{0}; j < 2; ++j) {
        if (const Symbol *last{GetLastSymbol(call.arguments[j])};
            last && !IsAllocatable(last->GetUltimate())) {
          context.messages().Say(call.arguments[j]->sourceLocation(),
              "Argument #%d to MOVE_ALLOC must be allocatable"_err_en_US,
              j + 1);
          ok = false;
        }
      }
      auto type0{call.arguments[0]->GetType()};
      auto type1{call.arguments[1]->GetType()};
      if (type0 && type1 && type0->IsPolymorphic() && !type1->IsPolymorphic()) {
        context.messages().Say(call.arguments[1]->sourceLocation(),
            "When MOVE_ALLOC(FROM=) is polymorphic, TO= must also be polymorphic"_err_en_US);
        ok = false;
      }
    }
  } else if (name == "present") {
    const auto &arg{call.arguments[0]};
    if (arg) {
      if (const auto *expr{arg->UnwrapExpr()}) {
        if (const Symbol *symbol{UnwrapWholeSymbolDataRef(*expr)}) {
          ok = symbol->attrs().test(semantics::Attr::OPTIONAL);
        }
      }
    }
    if (!ok) {
      context.messages().Say(
          arg ? arg->sourceLocation() : context.messages().at(),
          "Argument of PRESENT() must be the name of an OPTIONAL dummy argument"_err_en_US);
    }
  } else if (name == "reduce") { // 16.9.161
    std::optional<DynamicType> arrayType;
    if (const auto &array{call.arguments[0]}) {
      arrayType = array->GetType();
    }
    std::optional<characteristics::Procedure> procChars;
    parser::CharBlock at{context.messages().at()};
    if (const auto &operation{call.arguments[1]}) {
      if (const auto *expr{operation->UnwrapExpr()}) {
        if (const auto *designator{
                std::get_if<ProcedureDesignator>(&expr->u)}) {
          procChars =
              characteristics::Procedure::Characterize(*designator, context);
        } else if (const auto *ref{std::get_if<ProcedureRef>(&expr->u)}) {
          procChars = characteristics::Procedure::Characterize(*ref, context);
        }
      }
      if (auto operationAt{operation->sourceLocation()}) {
        at = *operationAt;
      }
    }
    if (!arrayType || !procChars) {
      ok = false; // error recovery
    } else {
      const auto *result{procChars->functionResult->GetTypeAndShape()};
      if (!procChars->IsPure() || procChars->dummyArguments.size() != 2 ||
          !procChars->functionResult) {
        ok = false;
        context.messages().Say(at,
            "OPERATION= argument of REDUCE() must be a pure function of two data arguments"_err_en_US);
      } else if (!result || result->Rank() != 0) {
        ok = false;
        context.messages().Say(at,
            "OPERATION= argument of REDUCE() must be a scalar function"_err_en_US);
      } else if (result->type().IsPolymorphic() ||
          !arrayType->IsTkLenCompatibleWith(result->type())) {
        ok = false;
        context.messages().Say(at,
            "OPERATION= argument of REDUCE() must have the same type as ARRAY="_err_en_US);
      } else {
        const characteristics::DummyDataObject *data[2]{};
        for (int j{0}; j < 2; ++j) {
          const auto &dummy{procChars->dummyArguments.at(j)};
          data[j] = std::get_if<characteristics::DummyDataObject>(&dummy.u);
          ok = ok && data[j];
        }
        if (!ok) {
          context.messages().Say(at,
              "OPERATION= argument of REDUCE() may not have dummy procedure arguments"_err_en_US);
        } else {
          for (int j{0}; j < 2; ++j) {
            ok = ok &&
                !data[j]->attrs.test(
                    characteristics::DummyDataObject::Attr::Optional) &&
                !data[j]->attrs.test(
                    characteristics::DummyDataObject::Attr::Allocatable) &&
                !data[j]->attrs.test(
                    characteristics::DummyDataObject::Attr::Pointer) &&
                data[j]->type.Rank() == 0 &&
                !data[j]->type.type().IsPolymorphic() &&
                data[j]->type.type().IsTkCompatibleWith(*arrayType);
          }
          if (!ok) {
            context.messages().Say(at,
                "Arguments of OPERATION= procedure of REDUCE() must be both scalar of the same type as ARRAY=, and neither allocatable, pointer, polymorphic, or optional"_err_en_US);
          } else if (data[0]->attrs.test(characteristics::DummyDataObject::
                             Attr::Asynchronous) !=
                  data[1]->attrs.test(
                      characteristics::DummyDataObject::Attr::Asynchronous) ||
              data[0]->attrs.test(
                  characteristics::DummyDataObject::Attr::Volatile) !=
                  data[1]->attrs.test(
                      characteristics::DummyDataObject::Attr::Volatile) ||
              data[0]->attrs.test(
                  characteristics::DummyDataObject::Attr::Target) !=
                  data[1]->attrs.test(
                      characteristics::DummyDataObject::Attr::Target)) {
            ok = false;
            context.messages().Say(at,
                "If either argument of the OPERATION= procedure of REDUCE() has the ASYNCHRONOUS, VOLATILE, or TARGET attribute, both must have that attribute"_err_en_US);
          }
        }
      }
    }
  } else if (name == "transfer") { // 16.9.193
    if (call.arguments.size() >= 2) {
      auto source{characteristics::TypeAndShape::Characterize(
          call.arguments[0], context)};
      auto mold{characteristics::TypeAndShape::Characterize(
          call.arguments[1], context)};
      if (source && mold && mold->Rank() > 0 &&
          evaluate::ToInt64(
              evaluate::Fold(
                  context, mold->MeasureElementSizeInBytes(context, false)))
                  .value_or(1) == 0) {
        if (auto sourceSize{evaluate::ToInt64(evaluate::Fold(
                context, source->MeasureSizeInBytes(context)))}) {
          if (*sourceSize > 0) {
            context.messages().Say(
                "Element size of MOLD= array may not be zero when SOURCE= is not empty"_err_en_US);
            ok = false;
          }
        } else {
          context.messages().Say(
              "Element size of MOLD= array may not be zero unless SOURCE= is empty"_warn_en_US);
        }
      }
      if (call.arguments.size() > 2) {
        if (const Symbol *whole{
                UnwrapWholeSymbolOrComponentDataRef(call.arguments[2])}) {
          if (IsOptional(*whole)) {
            context.messages().Say(
                "SIZE= argument may not be the optional dummy argument '%s'"_err_en_US,
                whole->name());
            ok = false;
          } else if (IsAllocatableOrPointer(*whole)) {
            context.messages().Say(
                "SIZE= argument that is allocatable or pointer must be present at execution; parenthesize to silence this warning"_warn_en_US);
          }
        }
      }
    }
  } else if (name == "ucobound") {
    return CheckDimAgainstCorank(call, context);
  }
  return ok;
}

static DynamicType GetReturnType(const SpecificIntrinsicInterface &interface,
    const common::IntrinsicTypeDefaultKinds &defaults) {
  TypeCategory category{TypeCategory::Integer};
  switch (interface.result.kindCode) {
  case KindCode::defaultIntegerKind:
    break;
  case KindCode::doublePrecision:
  case KindCode::defaultRealKind:
    category = TypeCategory::Real;
    break;
  default:
    CRASH_NO_CASE;
  }
  int kind{interface.result.kindCode == KindCode::doublePrecision
          ? defaults.doublePrecisionKind()
          : defaults.GetDefaultKind(category)};
  return DynamicType{category, kind};
}

// Probe the configured intrinsic procedure pattern tables in search of a
// match for a given procedure reference.
std::optional<SpecificCall> IntrinsicProcTable::Implementation::Probe(
    const CallCharacteristics &call, ActualArguments &arguments,
    FoldingContext &context) const {

  // All special cases handled here before the table probes below must
  // also be recognized as special names in IsIntrinsicSubroutine().
  if (call.isSubroutineCall) {
    if (call.name == "__builtin_c_f_pointer") {
      return HandleC_F_Pointer(arguments, context);
    } else if (call.name == "random_seed") {
      int optionalCount{0};
      for (const auto &arg : arguments) {
        if (const auto *expr{arg->UnwrapExpr()}) {
          optionalCount +=
              Fortran::evaluate::MayBePassedAsAbsentOptional(*expr, context);
        }
      }
      if (arguments.size() - optionalCount > 1) {
        context.messages().Say(
            "RANDOM_SEED must have either 1 or no arguments"_err_en_US);
      }
    }
  } else if (call.name == "null") {
    return HandleNull(arguments, context);
  }

  if (call.isSubroutineCall) {
    auto subrRange{subroutines_.equal_range(call.name)};
    for (auto iter{subrRange.first}; iter != subrRange.second; ++iter) {
      if (auto specificCall{iter->second->Match(
              call, defaults_, arguments, context, builtinsScope_)}) {
        ApplySpecificChecks(*specificCall, context);
        return specificCall;
      }
    }
    if (IsIntrinsicFunction(call.name)) {
      context.messages().Say(
          "Cannot use intrinsic function '%s' as a subroutine"_err_en_US,
          call.name);
    }
    return std::nullopt;
  }

  // Helper to avoid emitting errors before it is sure there is no match
  parser::Messages localBuffer;
  parser::Messages *finalBuffer{context.messages().messages()};
  parser::ContextualMessages localMessages{
      context.messages().at(), finalBuffer ? &localBuffer : nullptr};
  FoldingContext localContext{context, localMessages};
  auto matchOrBufferMessages{
      [&](const IntrinsicInterface &intrinsic,
          parser::Messages &buffer) -> std::optional<SpecificCall> {
        if (auto specificCall{intrinsic.Match(
                call, defaults_, arguments, localContext, builtinsScope_)}) {
          if (finalBuffer) {
            finalBuffer->Annex(std::move(localBuffer));
          }
          return specificCall;
        } else if (buffer.empty()) {
          buffer.Annex(std::move(localBuffer));
        } else {
          localBuffer.clear();
        }
        return std::nullopt;
      }};

  // Probe the generic intrinsic function table first; allow for
  // the use of a legacy alias.
  parser::Messages genericBuffer;
  const std::string &name{ResolveAlias(call.name)};
  auto genericRange{genericFuncs_.equal_range(name)};
  for (auto iter{genericRange.first}; iter != genericRange.second; ++iter) {
    if (auto specificCall{
            matchOrBufferMessages(*iter->second, genericBuffer)}) {
      ApplySpecificChecks(*specificCall, context);
      return specificCall;
    }
  }

  // Probe the specific intrinsic function table next.
  parser::Messages specificBuffer;
  auto specificRange{specificFuncs_.equal_range(call.name)};
  for (auto specIter{specificRange.first}; specIter != specificRange.second;
       ++specIter) {
    // We only need to check the cases with distinct generic names.
    if (const char *genericName{specIter->second->generic}) {
      if (auto specificCall{
              matchOrBufferMessages(*specIter->second, specificBuffer)}) {
        if (!specIter->second->useGenericAndForceResultType) {
          specificCall->specificIntrinsic.name = genericName;
        }
        specificCall->specificIntrinsic.isRestrictedSpecific =
            specIter->second->isRestrictedSpecific;
        // TODO test feature AdditionalIntrinsics, warn on nonstandard
        // specifics with DoublePrecisionComplex arguments.
        return specificCall;
      }
    }
  }

  // If there was no exact match with a specific, try to match the related
  // generic and convert the result to the specific required type.
  for (auto specIter{specificRange.first}; specIter != specificRange.second;
       ++specIter) {
    // We only need to check the cases with distinct generic names.
    if (const char *genericName{specIter->second->generic}) {
      if (specIter->second->useGenericAndForceResultType) {
        auto genericRange{genericFuncs_.equal_range(genericName)};
        for (auto genIter{genericRange.first}; genIter != genericRange.second;
             ++genIter) {
          if (auto specificCall{
                  matchOrBufferMessages(*genIter->second, specificBuffer)}) {
            // Force the call result type to the specific intrinsic result type
            DynamicType newType{GetReturnType(*specIter->second, defaults_)};
            context.messages().Say(
                "argument types do not match specific intrinsic '%s' "
                "requirements; using '%s' generic instead and converting the "
                "result to %s if needed"_port_en_US,
                call.name, genericName, newType.AsFortran());
            specificCall->specificIntrinsic.name = call.name;
            specificCall->specificIntrinsic.characteristics.value()
                .functionResult.value()
                .SetType(newType);
            return specificCall;
          }
        }
      }
    }
  }

  if (specificBuffer.empty() && genericBuffer.empty() &&
      IsIntrinsicSubroutine(call.name)) {
    context.messages().Say(
        "Cannot use intrinsic subroutine '%s' as a function"_err_en_US,
        call.name);
  }

  // No match; report the right errors, if any
  if (finalBuffer) {
    if (specificBuffer.empty()) {
      finalBuffer->Annex(std::move(genericBuffer));
    } else {
      finalBuffer->Annex(std::move(specificBuffer));
    }
  }
  return std::nullopt;
}

std::optional<SpecificIntrinsicFunctionInterface>
IntrinsicProcTable::Implementation::IsSpecificIntrinsicFunction(
    const std::string &name) const {
  auto specificRange{specificFuncs_.equal_range(name)};
  for (auto iter{specificRange.first}; iter != specificRange.second; ++iter) {
    const SpecificIntrinsicInterface &specific{*iter->second};
    std::string genericName{name};
    if (specific.generic) {
      genericName = std::string(specific.generic);
    }
    characteristics::FunctionResult fResult{GetSpecificType(specific.result)};
    characteristics::DummyArguments args;
    int dummies{specific.CountArguments()};
    for (int j{0}; j < dummies; ++j) {
      characteristics::DummyDataObject dummy{
          GetSpecificType(specific.dummy[j].typePattern)};
      dummy.intent = specific.dummy[j].intent;
      args.emplace_back(
          std::string{specific.dummy[j].keyword}, std::move(dummy));
    }
    characteristics::Procedure::Attrs attrs;
    attrs.set(characteristics::Procedure::Attr::Pure)
        .set(characteristics::Procedure::Attr::Elemental);
    characteristics::Procedure chars{
        std::move(fResult), std::move(args), attrs};
    return SpecificIntrinsicFunctionInterface{
        std::move(chars), genericName, specific.isRestrictedSpecific};
  }
  return std::nullopt;
}

DynamicType IntrinsicProcTable::Implementation::GetSpecificType(
    const TypePattern &pattern) const {
  const CategorySet &set{pattern.categorySet};
  CHECK(set.count() == 1);
  TypeCategory category{set.LeastElement().value()};
  if (pattern.kindCode == KindCode::doublePrecision) {
    return DynamicType{category, defaults_.doublePrecisionKind()};
  } else {
    return DynamicType{category, defaults_.GetDefaultKind(category)};
  }
}

IntrinsicProcTable::~IntrinsicProcTable() = default;

IntrinsicProcTable IntrinsicProcTable::Configure(
    const common::IntrinsicTypeDefaultKinds &defaults) {
  IntrinsicProcTable result;
  result.impl_ = std::make_unique<IntrinsicProcTable::Implementation>(defaults);
  return result;
}

void IntrinsicProcTable::SupplyBuiltins(
    const semantics::Scope &builtins) const {
  DEREF(impl_.get()).SupplyBuiltins(builtins);
}

bool IntrinsicProcTable::IsIntrinsic(const std::string &name) const {
  return DEREF(impl_.get()).IsIntrinsic(name);
}
bool IntrinsicProcTable::IsIntrinsicFunction(const std::string &name) const {
  return DEREF(impl_.get()).IsIntrinsicFunction(name);
}
bool IntrinsicProcTable::IsIntrinsicSubroutine(const std::string &name) const {
  return DEREF(impl_.get()).IsIntrinsicSubroutine(name);
}

IntrinsicClass IntrinsicProcTable::GetIntrinsicClass(
    const std::string &name) const {
  return DEREF(impl_.get()).GetIntrinsicClass(name);
}

std::string IntrinsicProcTable::GetGenericIntrinsicName(
    const std::string &name) const {
  return DEREF(impl_.get()).GetGenericIntrinsicName(name);
}

std::optional<SpecificCall> IntrinsicProcTable::Probe(
    const CallCharacteristics &call, ActualArguments &arguments,
    FoldingContext &context) const {
  return DEREF(impl_.get()).Probe(call, arguments, context);
}

std::optional<SpecificIntrinsicFunctionInterface>
IntrinsicProcTable::IsSpecificIntrinsicFunction(const std::string &name) const {
  return DEREF(impl_.get()).IsSpecificIntrinsicFunction(name);
}

llvm::raw_ostream &TypePattern::Dump(llvm::raw_ostream &o) const {
  if (categorySet == AnyType) {
    o << "any type";
  } else {
    const char *sep = "";
    auto set{categorySet};
    while (auto least{set.LeastElement()}) {
      o << sep << EnumToString(*least);
      sep = " or ";
      set.reset(*least);
    }
  }
  o << '(' << EnumToString(kindCode) << ')';
  return o;
}

llvm::raw_ostream &IntrinsicDummyArgument::Dump(llvm::raw_ostream &o) const {
  if (keyword) {
    o << keyword << '=';
  }
  return typePattern.Dump(o)
      << ' ' << EnumToString(rank) << ' ' << EnumToString(optionality)
      << EnumToString(intent);
}

llvm::raw_ostream &IntrinsicInterface::Dump(llvm::raw_ostream &o) const {
  o << name;
  char sep{'('};
  for (const auto &d : dummy) {
    if (d.typePattern.kindCode == KindCode::none) {
      break;
    }
    d.Dump(o << sep);
    sep = ',';
  }
  if (sep == '(') {
    o << "()";
  }
  return result.Dump(o << " -> ") << ' ' << EnumToString(rank);
}

llvm::raw_ostream &IntrinsicProcTable::Implementation::Dump(
    llvm::raw_ostream &o) const {
  o << "generic intrinsic functions:\n";
  for (const auto &iter : genericFuncs_) {
    iter.second->Dump(o << iter.first << ": ") << '\n';
  }
  o << "specific intrinsic functions:\n";
  for (const auto &iter : specificFuncs_) {
    iter.second->Dump(o << iter.first << ": ");
    if (const char *g{iter.second->generic}) {
      o << " -> " << g;
    }
    o << '\n';
  }
  o << "subroutines:\n";
  for (const auto &iter : subroutines_) {
    iter.second->Dump(o << iter.first << ": ") << '\n';
  }
  return o;
}

llvm::raw_ostream &IntrinsicProcTable::Dump(llvm::raw_ostream &o) const {
  return DEREF(impl_.get()).Dump(o);
}

// In general C846 prohibits allocatable coarrays to be passed to INTENT(OUT)
// dummy arguments. This rule does not apply to intrinsics in general.
// Some intrinsic explicitly allow coarray allocatable in their description.
// It is assumed that unless explicitly allowed for an intrinsic,
// this is forbidden.
// Since there are very few intrinsic identified that allow this, they are
// listed here instead of adding a field in the table.
bool AcceptsIntentOutAllocatableCoarray(const std::string &intrinsic) {
  return intrinsic == "move_alloc";
}
} // namespace Fortran::evaluate
