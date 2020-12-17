//===-- include/flang/Evaluate/type.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_TYPE_H_
#define FORTRAN_EVALUATE_TYPE_H_

// These definitions map Fortran's intrinsic types, characterized by byte
// sizes encoded in KIND type parameter values, to their value representation
// types in the evaluation library, which are parameterized in terms of
// total bit width and real precision.  Instances of the Type class template
// are suitable for use as template parameters to instantiate other class
// templates, like expressions, over the supported types and kinds.

#include "common.h"
#include "complex.h"
#include "formatting.h"
#include "integer.h"
#include "logical.h"
#include "real.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/idioms.h"
#include "flang/Common/real.h"
#include "flang/Common/template.h"
#include <cinttypes>
#include <optional>
#include <string>
#include <type_traits>
#include <variant>

namespace Fortran::semantics {
class DeclTypeSpec;
class DerivedTypeSpec;
class ParamValue;
class Symbol;
bool IsDescriptor(const Symbol &);
} // namespace Fortran::semantics

namespace Fortran::evaluate {

using common::TypeCategory;

// Specific intrinsic types are represented by specializations of
// this class template Type<CATEGORY, KIND>.
template <TypeCategory CATEGORY, int KIND = 0> class Type;

using SubscriptInteger = Type<TypeCategory::Integer, 8>;
using CInteger = Type<TypeCategory::Integer, 4>;
using LogicalResult = Type<TypeCategory::Logical, 4>;
using LargestReal = Type<TypeCategory::Real, 16>;

// A predicate that is true when a kind value is a kind that could possibly
// be supported for an intrinsic type category on some target instruction
// set architecture.
// TODO: specialize for the actual target architecture
static constexpr bool IsValidKindOfIntrinsicType(
    TypeCategory category, std::int64_t kind) {
  switch (category) {
  case TypeCategory::Integer:
    return kind == 1 || kind == 2 || kind == 4 || kind == 8 || kind == 16;
  case TypeCategory::Real:
  case TypeCategory::Complex:
    return kind == 2 || kind == 3 || kind == 4 || kind == 8 || kind == 10 ||
        kind == 16;
  case TypeCategory::Character:
    return kind == 1 || kind == 2 || kind == 4;
  case TypeCategory::Logical:
    return kind == 1 || kind == 2 || kind == 4 || kind == 8;
  default:
    return false;
  }
}

// DynamicType is meant to be suitable for use as the result type for
// GetType() functions and member functions; consequently, it must be
// capable of being used in a constexpr context.  So it does *not*
// directly hold anything requiring a destructor, such as an arbitrary
// CHARACTER length type parameter expression.  Those must be derived
// via LEN() member functions, packaged elsewhere (e.g. as in
// ArrayConstructor), or copied from a parameter spec in the symbol table
// if one is supplied.
class DynamicType {
public:
  constexpr DynamicType(TypeCategory cat, int k) : category_{cat}, kind_{k} {
    CHECK(IsValidKindOfIntrinsicType(category_, kind_));
  }
  constexpr DynamicType(int k, const semantics::ParamValue &pv)
      : category_{TypeCategory::Character}, kind_{k}, charLength_{&pv} {
    CHECK(IsValidKindOfIntrinsicType(category_, kind_));
  }
  explicit constexpr DynamicType(
      const semantics::DerivedTypeSpec &dt, bool poly = false)
      : category_{TypeCategory::Derived}, derived_{&dt} {
    if (poly) {
      kind_ = ClassKind;
    }
  }
  CONSTEXPR_CONSTRUCTORS_AND_ASSIGNMENTS(DynamicType)

  // A rare use case used for representing the characteristics of an
  // intrinsic function like REAL() that accepts a typeless BOZ literal
  // argument and for typeless pointers -- things that real user Fortran can't
  // do.
  static constexpr DynamicType TypelessIntrinsicArgument() {
    DynamicType result;
    result.category_ = TypeCategory::Integer;
    result.kind_ = TypelessKind;
    return result;
  }

  static constexpr DynamicType UnlimitedPolymorphic() {
    DynamicType result;
    result.category_ = TypeCategory::Derived;
    result.kind_ = ClassKind;
    result.derived_ = nullptr;
    return result; // CLASS(*)
  }

  static constexpr DynamicType AssumedType() {
    DynamicType result;
    result.category_ = TypeCategory::Derived;
    result.kind_ = AssumedTypeKind;
    result.derived_ = nullptr;
    return result; // TYPE(*)
  }

  // Comparison is deep -- type parameters are compared independently.
  bool operator==(const DynamicType &) const;
  bool operator!=(const DynamicType &that) const { return !(*this == that); }

  constexpr TypeCategory category() const { return category_; }
  constexpr int kind() const {
    CHECK(kind_ > 0);
    return kind_;
  }
  constexpr const semantics::ParamValue *charLength() const {
    return charLength_;
  }
  std::optional<Expr<SubscriptInteger>> GetCharLength() const;

  std::size_t GetAlignment(const FoldingContext &) const;
  std::optional<Expr<SubscriptInteger>> MeasureSizeInBytes(
      FoldingContext &, bool aligned) const;

  std::string AsFortran() const;
  std::string AsFortran(std::string &&charLenExpr) const;
  DynamicType ResultTypeForMultiply(const DynamicType &) const;

  bool IsAssumedLengthCharacter() const;
  bool IsNonConstantLengthCharacter() const;
  bool IsTypelessIntrinsicArgument() const;
  constexpr bool IsAssumedType() const { // TYPE(*)
    return kind_ == AssumedTypeKind;
  }
  constexpr bool IsPolymorphic() const { // TYPE(*) or CLASS()
    return kind_ == ClassKind || IsAssumedType();
  }
  constexpr bool IsUnlimitedPolymorphic() const { // TYPE(*) or CLASS(*)
    return IsPolymorphic() && !derived_;
  }
  constexpr const semantics::DerivedTypeSpec &GetDerivedTypeSpec() const {
    return DEREF(derived_);
  }

  bool RequiresDescriptor() const;
  bool HasDeferredTypeParameter() const;

  // 7.3.2.3 & 15.5.2.4 type compatibility.
  // x.IsTkCompatibleWith(y) is true if "x => y" or passing actual y to
  // dummy argument x would be valid.  Be advised, this is not a reflexive
  // relation.  Kind type parameters must match.
  bool IsTkCompatibleWith(const DynamicType &) const;

  // Result will be missing when a symbol is absent or
  // has an erroneous type, e.g., REAL(KIND=666).
  static std::optional<DynamicType> From(const semantics::DeclTypeSpec &);
  static std::optional<DynamicType> From(const semantics::Symbol &);

  template <typename A> static std::optional<DynamicType> From(const A &x) {
    return x.GetType();
  }
  template <typename A> static std::optional<DynamicType> From(const A *p) {
    if (!p) {
      return std::nullopt;
    } else {
      return From(*p);
    }
  }
  template <typename A>
  static std::optional<DynamicType> From(const std::optional<A> &x) {
    if (x) {
      return From(*x);
    } else {
      return std::nullopt;
    }
  }

private:
  // Special kind codes are used to distinguish the following Fortran types.
  enum SpecialKind {
    TypelessKind = -1, // BOZ actual argument to intrinsic function or pointer
                       // argument to ASSOCIATED
    ClassKind = -2, // CLASS(T) or CLASS(*)
    AssumedTypeKind = -3, // TYPE(*)
  };

  constexpr DynamicType() {}

  TypeCategory category_{TypeCategory::Derived}; // overridable default
  int kind_{0};
  const semantics::ParamValue *charLength_{nullptr};
  const semantics::DerivedTypeSpec *derived_{nullptr}; // TYPE(T), CLASS(T)
};

// Return the DerivedTypeSpec of a DynamicType if it has one.
const semantics::DerivedTypeSpec *GetDerivedTypeSpec(const DynamicType &);
const semantics::DerivedTypeSpec *GetDerivedTypeSpec(
    const std::optional<DynamicType> &);
const semantics::DerivedTypeSpec *GetParentTypeSpec(
    const semantics::DerivedTypeSpec &);

std::string DerivedTypeSpecAsFortran(const semantics::DerivedTypeSpec &);

template <TypeCategory CATEGORY, int KIND = 0> struct TypeBase {
  static constexpr TypeCategory category{CATEGORY};
  static constexpr int kind{KIND};
  constexpr bool operator==(const TypeBase &) const { return true; }
  static constexpr DynamicType GetType() { return {category, kind}; }
  static std::string AsFortran() { return GetType().AsFortran(); }
};

template <int KIND>
class Type<TypeCategory::Integer, KIND>
    : public TypeBase<TypeCategory::Integer, KIND> {
public:
  using Scalar = value::Integer<8 * KIND>;
};

template <int KIND>
class Type<TypeCategory::Real, KIND>
    : public TypeBase<TypeCategory::Real, KIND> {
public:
  static constexpr int precision{common::PrecisionOfRealKind(KIND)};
  static constexpr int bits{common::BitsForBinaryPrecision(precision)};
  using Scalar = value::Real<value::Integer<bits>, precision>;
};

// The KIND type parameter on COMPLEX is the kind of each of its components.
template <int KIND>
class Type<TypeCategory::Complex, KIND>
    : public TypeBase<TypeCategory::Complex, KIND> {
public:
  using Part = Type<TypeCategory::Real, KIND>;
  using Scalar = value::Complex<typename Part::Scalar>;
};

template <>
class Type<TypeCategory::Character, 1>
    : public TypeBase<TypeCategory::Character, 1> {
public:
  using Scalar = std::string;
};

template <>
class Type<TypeCategory::Character, 2>
    : public TypeBase<TypeCategory::Character, 2> {
public:
  using Scalar = std::u16string;
};

template <>
class Type<TypeCategory::Character, 4>
    : public TypeBase<TypeCategory::Character, 4> {
public:
  using Scalar = std::u32string;
};

template <int KIND>
class Type<TypeCategory::Logical, KIND>
    : public TypeBase<TypeCategory::Logical, KIND> {
public:
  using Scalar = value::Logical<8 * KIND>;
};

// Type functions

// Given a specific type, find the type of the same kind in another category.
template <TypeCategory CATEGORY, typename T>
using SameKind = Type<CATEGORY, std::decay_t<T>::kind>;

// Many expressions, including subscripts, CHARACTER lengths, array bounds,
// and effective type parameter values, are of a maximal kind of INTEGER.
using IndirectSubscriptIntegerExpr =
    common::CopyableIndirection<Expr<SubscriptInteger>>;

// For each intrinsic type category CAT, CategoryTypes<CAT> is an instantiation
// of std::tuple<Type<CAT, K>> that comprises every kind value K in that
// category that could possibly be supported on any target.
template <TypeCategory CATEGORY, int KIND>
using CategoryKindTuple =
    std::conditional_t<IsValidKindOfIntrinsicType(CATEGORY, KIND),
        std::tuple<Type<CATEGORY, KIND>>, std::tuple<>>;

template <TypeCategory CATEGORY, int... KINDS>
using CategoryTypesHelper =
    common::CombineTuples<CategoryKindTuple<CATEGORY, KINDS>...>;

template <TypeCategory CATEGORY>
using CategoryTypes = CategoryTypesHelper<CATEGORY, 1, 2, 3, 4, 8, 10, 16, 32>;

using IntegerTypes = CategoryTypes<TypeCategory::Integer>;
using RealTypes = CategoryTypes<TypeCategory::Real>;
using ComplexTypes = CategoryTypes<TypeCategory::Complex>;
using CharacterTypes = CategoryTypes<TypeCategory::Character>;
using LogicalTypes = CategoryTypes<TypeCategory::Logical>;

using FloatingTypes = common::CombineTuples<RealTypes, ComplexTypes>;
using NumericTypes = common::CombineTuples<IntegerTypes, FloatingTypes>;
using RelationalTypes = common::CombineTuples<NumericTypes, CharacterTypes>;
using AllIntrinsicTypes = common::CombineTuples<RelationalTypes, LogicalTypes>;
using LengthlessIntrinsicTypes =
    common::CombineTuples<NumericTypes, LogicalTypes>;

// Predicates: does a type represent a specific intrinsic type?
template <typename T>
constexpr bool IsSpecificIntrinsicType{common::HasMember<T, AllIntrinsicTypes>};

// Predicate: is a type an intrinsic type that is completely characterized
// by its category and kind parameter value, or might it have a derived type
// &/or a length type parameter?
template <typename T>
constexpr bool IsLengthlessIntrinsicType{
    common::HasMember<T, LengthlessIntrinsicTypes>};

// Represents a type of any supported kind within a particular category.
template <TypeCategory CATEGORY> struct SomeKind {
  static constexpr TypeCategory category{CATEGORY};
  constexpr bool operator==(const SomeKind &) const { return true; }
};

using NumericCategoryTypes = std::tuple<SomeKind<TypeCategory::Integer>,
    SomeKind<TypeCategory::Real>, SomeKind<TypeCategory::Complex>>;
using AllIntrinsicCategoryTypes = std::tuple<SomeKind<TypeCategory::Integer>,
    SomeKind<TypeCategory::Real>, SomeKind<TypeCategory::Complex>,
    SomeKind<TypeCategory::Character>, SomeKind<TypeCategory::Logical>>;

// Represents a completely generic type (or, for Expr<SomeType>, a typeless
// value like a BOZ literal or NULL() pointer).
struct SomeType {};

class StructureConstructor;

// Represents any derived type, polymorphic or not, as well as CLASS(*).
template <> class SomeKind<TypeCategory::Derived> {
public:
  static constexpr TypeCategory category{TypeCategory::Derived};
  using Scalar = StructureConstructor;

  constexpr SomeKind() {} // CLASS(*)
  constexpr explicit SomeKind(const semantics::DerivedTypeSpec &dts)
      : derivedTypeSpec_{&dts} {}
  constexpr explicit SomeKind(const DynamicType &dt)
      : SomeKind(dt.GetDerivedTypeSpec()) {}
  CONSTEXPR_CONSTRUCTORS_AND_ASSIGNMENTS(SomeKind)

  bool IsUnlimitedPolymorphic() const { return !derivedTypeSpec_; }
  constexpr DynamicType GetType() const {
    if (!derivedTypeSpec_) {
      return DynamicType::UnlimitedPolymorphic();
    } else {
      return DynamicType{*derivedTypeSpec_};
    }
  }
  const semantics::DerivedTypeSpec &derivedTypeSpec() const {
    CHECK(derivedTypeSpec_);
    return *derivedTypeSpec_;
  }
  bool operator==(const SomeKind &) const;
  std::string AsFortran() const;

private:
  const semantics::DerivedTypeSpec *derivedTypeSpec_{nullptr};
};

using SomeInteger = SomeKind<TypeCategory::Integer>;
using SomeReal = SomeKind<TypeCategory::Real>;
using SomeComplex = SomeKind<TypeCategory::Complex>;
using SomeCharacter = SomeKind<TypeCategory::Character>;
using SomeLogical = SomeKind<TypeCategory::Logical>;
using SomeDerived = SomeKind<TypeCategory::Derived>;
using SomeCategory = std::tuple<SomeInteger, SomeReal, SomeComplex,
    SomeCharacter, SomeLogical, SomeDerived>;

using AllTypes =
    common::CombineTuples<AllIntrinsicTypes, std::tuple<SomeDerived>>;

template <typename T> using Scalar = typename std::decay_t<T>::Scalar;

// When Scalar<T> is S, then TypeOf<S> is T.
// TypeOf is implemented by scanning all supported types for a match
// with Type<T>::Scalar.
template <typename CONST> struct TypeOfHelper {
  template <typename T> struct Predicate {
    static constexpr bool value() {
      return std::is_same_v<std::decay_t<CONST>,
          std::decay_t<typename T::Scalar>>;
    }
  };
  static constexpr int index{
      common::SearchMembers<Predicate, AllIntrinsicTypes>};
  using type = std::conditional_t<index >= 0,
      std::tuple_element_t<index, AllIntrinsicTypes>, void>;
};

template <typename CONST> using TypeOf = typename TypeOfHelper<CONST>::type;

int SelectedCharKind(const std::string &, int defaultKind);
int SelectedIntKind(std::int64_t precision = 0);
int SelectedRealKind(
    std::int64_t precision = 0, std::int64_t range = 0, std::int64_t radix = 2);

// For generating "[extern] template class", &c. boilerplate
#define EXPAND_FOR_EACH_INTEGER_KIND(M, P, S) \
  M(P, S, 1) M(P, S, 2) M(P, S, 4) M(P, S, 8) M(P, S, 16)
#define EXPAND_FOR_EACH_REAL_KIND(M, P, S) \
  M(P, S, 2) M(P, S, 3) M(P, S, 4) M(P, S, 8) M(P, S, 10) M(P, S, 16)
#define EXPAND_FOR_EACH_COMPLEX_KIND(M, P, S) EXPAND_FOR_EACH_REAL_KIND(M, P, S)
#define EXPAND_FOR_EACH_CHARACTER_KIND(M, P, S) M(P, S, 1) M(P, S, 2) M(P, S, 4)
#define EXPAND_FOR_EACH_LOGICAL_KIND(M, P, S) \
  M(P, S, 1) M(P, S, 2) M(P, S, 4) M(P, S, 8)
#define TEMPLATE_INSTANTIATION(P, S, ARG) P<ARG> S;

#define FOR_EACH_INTEGER_KIND_HELP(PREFIX, SUFFIX, K) \
  PREFIX<Type<TypeCategory::Integer, K>> SUFFIX;
#define FOR_EACH_REAL_KIND_HELP(PREFIX, SUFFIX, K) \
  PREFIX<Type<TypeCategory::Real, K>> SUFFIX;
#define FOR_EACH_COMPLEX_KIND_HELP(PREFIX, SUFFIX, K) \
  PREFIX<Type<TypeCategory::Complex, K>> SUFFIX;
#define FOR_EACH_CHARACTER_KIND_HELP(PREFIX, SUFFIX, K) \
  PREFIX<Type<TypeCategory::Character, K>> SUFFIX;
#define FOR_EACH_LOGICAL_KIND_HELP(PREFIX, SUFFIX, K) \
  PREFIX<Type<TypeCategory::Logical, K>> SUFFIX;

#define FOR_EACH_INTEGER_KIND(PREFIX, SUFFIX) \
  EXPAND_FOR_EACH_INTEGER_KIND(FOR_EACH_INTEGER_KIND_HELP, PREFIX, SUFFIX)
#define FOR_EACH_REAL_KIND(PREFIX, SUFFIX) \
  EXPAND_FOR_EACH_REAL_KIND(FOR_EACH_REAL_KIND_HELP, PREFIX, SUFFIX)
#define FOR_EACH_COMPLEX_KIND(PREFIX, SUFFIX) \
  EXPAND_FOR_EACH_COMPLEX_KIND(FOR_EACH_COMPLEX_KIND_HELP, PREFIX, SUFFIX)
#define FOR_EACH_CHARACTER_KIND(PREFIX, SUFFIX) \
  EXPAND_FOR_EACH_CHARACTER_KIND(FOR_EACH_CHARACTER_KIND_HELP, PREFIX, SUFFIX)
#define FOR_EACH_LOGICAL_KIND(PREFIX, SUFFIX) \
  EXPAND_FOR_EACH_LOGICAL_KIND(FOR_EACH_LOGICAL_KIND_HELP, PREFIX, SUFFIX)

#define FOR_EACH_LENGTHLESS_INTRINSIC_KIND(PREFIX, SUFFIX) \
  FOR_EACH_INTEGER_KIND(PREFIX, SUFFIX) \
  FOR_EACH_REAL_KIND(PREFIX, SUFFIX) \
  FOR_EACH_COMPLEX_KIND(PREFIX, SUFFIX) \
  FOR_EACH_LOGICAL_KIND(PREFIX, SUFFIX)
#define FOR_EACH_INTRINSIC_KIND(PREFIX, SUFFIX) \
  FOR_EACH_LENGTHLESS_INTRINSIC_KIND(PREFIX, SUFFIX) \
  FOR_EACH_CHARACTER_KIND(PREFIX, SUFFIX)
#define FOR_EACH_SPECIFIC_TYPE(PREFIX, SUFFIX) \
  FOR_EACH_INTRINSIC_KIND(PREFIX, SUFFIX) \
  PREFIX<SomeDerived> SUFFIX;

#define FOR_EACH_CATEGORY_TYPE(PREFIX, SUFFIX) \
  PREFIX<SomeInteger> SUFFIX; \
  PREFIX<SomeReal> SUFFIX; \
  PREFIX<SomeComplex> SUFFIX; \
  PREFIX<SomeCharacter> SUFFIX; \
  PREFIX<SomeLogical> SUFFIX; \
  PREFIX<SomeDerived> SUFFIX; \
  PREFIX<SomeType> SUFFIX;
#define FOR_EACH_TYPE_AND_KIND(PREFIX, SUFFIX) \
  FOR_EACH_INTRINSIC_KIND(PREFIX, SUFFIX) \
  FOR_EACH_CATEGORY_TYPE(PREFIX, SUFFIX)
} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_TYPE_H_
