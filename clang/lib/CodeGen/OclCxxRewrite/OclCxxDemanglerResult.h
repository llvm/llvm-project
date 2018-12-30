//===- OclCxxDemanglerResult.h - OCLC++ demangler result object -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//
//
// Copyright (c) 2015 The Khronos Group Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and/or associated documentation files (the
// "Materials"), to deal in the Materials without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Materials, and to
// permit persons to whom the Materials are furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be included
// in all copies or substantial portions of the Materials.
//
// THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
//
//===----------------------------------------------------------------------===//


#ifndef CLANG_LIB_CODEGEN_OCLCXXREWRITE_OCLCXXDEMANGLERRESULT_H
#define CLANG_LIB_CODEGEN_OCLCXXREWRITE_OCLCXXDEMANGLERRESULT_H

#include <cassert>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>
#include <utility>

namespace oclcxx {
namespace adaptation {

// -----------------------------------------------------------------------------
// HELPERS FOR RESULT OBJECTS
// -----------------------------------------------------------------------------

/// Helper entity for returning empty string by const reference.
extern const std::string DmngRsltEmptyString;

/// \brief Iterator range helper class (for const_iterator).
///
/// Helper class which allows to iterate over range of iterators using
/// for-each C++ construct.
template <typename ConstItT>
class DmngRsltConstItRange {
public:
  using iterator = ConstItT;
  using const_iterator = ConstItT;

  const_iterator begin() const { return Begin; }
  const_iterator end() const { return End; }

  const_iterator cbegin() const { return Begin; }
  const_iterator cend() const { return End; }


  DmngRsltConstItRange(const const_iterator& Begin, const const_iterator& End)
    : Begin(Begin), End(End) {}


private:
  const_iterator Begin;
  const_iterator End;
};

/// \brief SFINAE enabler for sub-node (shared) pointers passed as paramaters.
///
/// This version is for parameters using std::forward pattern.
template <typename ValueT, typename TargetNodeT>
using SubnodeEnablerT = typename std::enable_if<
  std::is_constructible<std::shared_ptr<TargetNodeT>, ValueT &&>::value,
  void>::type;

/// \brief Helper class that creates type-dependant type which always returns
///        false.
///
/// Used in static_assert to force late evalution of condition.
template <typename ... ValuesT>
  struct AlwaysFalse : std::integral_constant<bool, false> {};


// -----------------------------------------------------------------------------
// ENUMERATIONS FOR RESULT OBJECTS
// -----------------------------------------------------------------------------

/// \brief Kind of node in demangled name (node category).
enum DmngNodeKind {
  DNDK_Name,
  DNDK_NamePart,
  DNDK_Type,
  DNDK_Expr,
  DNDK_NameParts,
};

/// \brief Kind of expression in demangled name.
enum DmngExprKind {
  DXK_TemplateParam,   ///< Template parameter expression.
  DXK_Decltype,        ///< decltype() expression.
  DXK_Primary,         ///< Primary expression (literals).
};

/// \brief Code that identifies additional operator name in expression.
enum DmngExprOpName {
  DXON_SuffixPlusPlus,     ///<                    # (E++)
  DXON_SuffixMinusMinus,   ///<                    # (E--)
  DXON_PrefixPlusPlus,     ///<                    # (++E)
  DXON_PrefixMinusMinus,   ///<                    # (--E)
};


/// \brief Kind of demangled types in result.
enum DmngTypeKind {
  DTK_Builtin,                 ///< Built-in types.
  // NOTE: It is not substitution when it is component of pointer 2 member.
  DTK_Function,                ///< Function types.
  /// Dependent and non-dependent type names or dependent typename-specifiers
  /// (optionally elaborated with specific keyword).
  DTK_TypeName,
  DTK_Array,                   ///< Array types (including variable size).
  DTK_Vector,                  ///< Vector types.
  DTK_PointerToMember,         ///< Pointer to member type.
  DTK_TemplateParam,           ///< Template parameters.
  DTK_Decltype,                ///< decltype() expressions.

  DTK_Pointer,                 ///< Pointer to type (E *).
  DTK_LValueRef,               ///< l-value reference to type (E &).
  DTK_RValueRef,               ///< r-value reference to type (E &&).
  DTK_C2000Complex,            ///< C2000 complex (complex E).
  DTK_C2000Imaginary,          ///< C2000 imaginary (imaginary E).
  DTK_PackExpansion,           ///< Pack expansion of type.

  /// Type qualified with set of qualifiers (cvr-quals, address space, vendor).
  DTK_QualGroup
};

/// \brief Type name kind (elaboration).
enum DmngTypeNameKind {
  DTNK_None,             ///< No elaboration.
  DTNK_ElaboratedClass,  ///< Elaborated with "class" or "struct" keyword.
  DTNK_ElaboratedUnion,  ///< Elaborated with "union" keyword.
  DTNK_ElaboratedEnum    ///< Elaborated with "enum" keyword.
};

/// Enumeration that describes built-in types in mangling.
enum DmngBuiltinType {
#define OCLCXX_MENC_BITYPE_FIXED(name, encName, cxxName) DBT_##name,
#include "OclCxxMangleEncodings.inc"
#undef OCLCXX_MENC_BITYPE_FIXED

#define OCLCXX_MENC_BITYPE(name) DBT_##name,
#include "OclCxxMangleEncodings.inc"
#undef OCLCXX_MENC_BITYPE

#define OCLCXX_MENC_BITYPE_ALIAS(aliasName, origName) \
  DBT_##aliasName = DBT_##origName,
#include "OclCxxMangleEncodings.inc"
#undef OCLCXX_MENC_BITYPE_ALIAS
};


/// \brief Kind of demangled name.
enum DmngNameKind {
  /// Name referes to ordinary data (global variable, etc.) or function
  /// (function, member function, etc.).
  DNK_Ordinary,
  /// Special names (virtual table entries, VTT, type_info, etc.).
  /// They have sub-kind: DmngSpecialNameKind
  DNK_Special
};

/// \brief Sub-kind of special demangled names.
enum DmngSpecialNameKind {
  /// Virtual type for specific. type (data structure with pointers
  /// to functions).
  DSNK_VirtualTable,
  /// Virtual table that describes adjustments for virtual tables in cases
  /// of muliple inheritence (data structure which points to another virtual
  /// tables generated for object-base object relations).
  DSNK_VirtualTableTable,
  /// Data structure with type information for specific type.
  DSNK_TypeInfoStruct,
  /// Global variable (data) with name string that identifies specific type.
  DSNK_TypeInfoNameString,
  /// Special function which adjust "this" and delagtes call to the
  /// real function.
  DSNK_VirtualThunk,
  /// Guard variable for one-time initialization of (inline) static object.
  DSNK_GuardVariable,
  /// Temporary variable which is bound (directly or indirectly) to reference
  /// that have static storage duration and, because of that, life extended to
  /// static storage duration as well.
  DSNK_LifeExtTemporary
};

/// \brief cvr-qualifiers (const, volatile, restrict or any mix of thereof).
///
/// This is flags enum and has bitwise operation provided.
enum DmngCvrQuals {
  DCVQ_None     = 0, ///< (no qualifiers)
  DCVQ_Const    = 1, ///< const
  DCVQ_Volatile = 2, ///< volatile
  DCVQ_Restrict = 4  ///< restrict (C99 extension)
};

inline DmngCvrQuals operator |(DmngCvrQuals LHS, DmngCvrQuals RHS) {
  using UTypeT = std::underlying_type<DmngCvrQuals>::type;

  return static_cast<DmngCvrQuals>(
    static_cast<UTypeT>(LHS) | static_cast<UTypeT>(RHS));
}

inline DmngCvrQuals operator &(DmngCvrQuals LHS, DmngCvrQuals RHS) {
  using UTypeT = std::underlying_type<DmngCvrQuals>::type;

  return static_cast<DmngCvrQuals>(
    static_cast<UTypeT>(LHS) & static_cast<UTypeT>(RHS));
}

inline DmngCvrQuals operator ^(DmngCvrQuals LHS, DmngCvrQuals RHS) {
  using UTypeT = std::underlying_type<DmngCvrQuals>::type;

  return static_cast<DmngCvrQuals>(
    static_cast<UTypeT>(LHS) ^ static_cast<UTypeT>(RHS));
}

inline DmngCvrQuals operator ~(DmngCvrQuals RHS) {
  using UTypeT = std::underlying_type<DmngCvrQuals>::type;
  const DmngCvrQuals FullSet = DCVQ_Const | DCVQ_Volatile | DCVQ_Restrict;

  return static_cast<DmngCvrQuals>(
    static_cast<UTypeT>(FullSet) ^ static_cast<UTypeT>(RHS));
}

/// \brief Reference qualifiers (for functions and references).
enum DmngRefQuals {
  DRQ_None,      ///< (none).
  DRQ_LValueRef, ///< &
  DRQ_RValueRef  ///< &&
};

/// \brief Address space qualifiers (clang mangling extension).
///
/// Result from parsing vendor-extended mangling in form: "U3AS[0-9]".
enum DmngAddressSpaceQuals {
  DASQ_None,     ///< (none).
  DASQ_Private,  ///< __private
  DASQ_Local,    ///< __local
  DASQ_Global,   ///< __global
  DASQ_Constant, ///< __constant
  DASQ_Generic   ///< __generic
};

/// \brief Kind of name part (one or multiple name parts form name
///        or nested name).
enum DmngNamePartKind {
  DNPK_Operator,         ///< Operators (operator +, etc.).
  DNPK_Constructor,      ///< Constructors (S::S(), etc.).
  DNPK_Destructor,       ///< Destructors  (S::~S(), etc.).
  DNPK_Source,           ///< Source names (namespace and class names, etc.).
  DNPK_UnnamedType,      ///< Unnamed types (closure/lambda classes, etc.).
  DNPK_TemplateParam,    ///< Template parameters (inherit from template arg).
  DNPK_Decltype,         ///< decltype(...) elements.
  DNPK_DataMember        ///< Data member initializers.
};

/// \brief Code that identifies operator name.
enum DmngOperatorName {
#define OCLCXX_MENC_OPR_FIXED(name, encName, arity, cxxName) DON_##name,
#include "OclCxxMangleEncodings.inc"
#undef OCLCXX_MENC_OPR_FIXED

#define OCLCXX_MENC_OPR(name, arity) DON_##name,
#include "OclCxxMangleEncodings.inc"
#undef OCLCXX_MENC_OPR

#define OCLCXX_MENC_OPR_ALIAS(aliasName, origName) \
  DON_##aliasName = DON_##origName,
#include "OclCxxMangleEncodings.inc"
#undef OCLCXX_MENC_OPR_ALIAS
};

/// \brief Type of constructor or destructor.
enum DmngCtorDtorType {
  DCDT_BaseObj,     ///< Base object constructor or destructor.
  DCDT_CompleteObj, ///< Complete object constructor or destructor.
  DCDT_DynMemObj    ///< Complete object allocating ctor or deleting dtor.
};


// -----------------------------------------------------------------------------
// FORWARD DECLARATIONS FOR RESULT OBJECTS
// -----------------------------------------------------------------------------

class DmngRsltNode;

class DmngRsltExpr;
class DmngRsltDecltypeExpr;
class DmngRsltTParamExpr;
class DmngRsltPrimaryExpr;

class DmngRsltType;
class DmngRsltBuiltinType;
class DmngRsltFuncType;
class DmngRsltTypeNameType;
class DmngRsltArrayVecType;
class DmngRsltPtr2MmbrType;
class DmngRsltTParamType;
class DmngRsltDecltypeType;
class DmngRsltQualType;
class DmngRsltQualGrpType;

class DmngRsltNamePart;
class DmngRsltOpNamePart;
class DmngRsltCtorDtorNamePart;
class DmngRsltSrcNamePart;
class DmngRsltUnmTypeNamePart;
class DmngRsltTParamNamePart;
class DmngRsltDecltypeNamePart;

class DmngRsltNameParts;

class DmngRsltName;
class DmngRsltOrdinaryName;
class DmngRsltSpecialName;


// -----------------------------------------------------------------------------
// TYPE SELECTORS FOR RESULT OBJECTS
// -----------------------------------------------------------------------------

/// \brief Kind selector helper for derivates of DmngRsltNode.
///
/// It allows to select derived type from DmngRsltNode based on node kind.
///
/// \tparam Kind Kind of derived type which will be selected.
template <DmngNodeKind Kind>
struct DmngRsltNodeKSel {
  /// Type selected by kind selector (fall-back: base type).
  using Type = DmngRsltNode;
};

template <>
struct DmngRsltNodeKSel<DNDK_Name> {
  using Type = DmngRsltName;
};

template <>
struct DmngRsltNodeKSel<DNDK_NamePart> {
  using Type = DmngRsltNamePart;
};

template <>
struct DmngRsltNodeKSel<DNDK_NameParts> {
  using Type = DmngRsltNameParts;
};

template <>
struct DmngRsltNodeKSel<DNDK_Type> {
  using Type = DmngRsltType;
};

template <>
struct DmngRsltNodeKSel<DNDK_Expr> {
  using Type = DmngRsltExpr;
};

template <DmngNodeKind Kind>
using DmngRsltNodeKSelT = typename DmngRsltNodeKSel<Kind>::Type;


/// \brief Kind selector helper for derivates of DmngRsltName.
///
/// It allows to select derived type from DmngRsltName based on name kind.
///
/// \tparam Kind Kind of derived type which will be selected.
template <DmngNameKind Kind>
struct DmngRsltNameKSel {
  /// Type selected by kind selector (fall-back: base type).
  using Type = DmngRsltName;
};

template <>
struct DmngRsltNameKSel<DNK_Ordinary> {
  using Type = DmngRsltOrdinaryName;
};

template <>
struct DmngRsltNameKSel<DNK_Special> {
  using Type = DmngRsltSpecialName;
};

template <DmngNameKind Kind>
using DmngRsltNameKSelT = typename DmngRsltNameKSel<Kind>::Type;


/// \brief Kind selector helper for derivates of DmngRsltNamePart.
///
/// It allows to select derived type from DmngRsltNamePart based on name kind.
///
/// \tparam Kind Kind of derived type which will be selected.
template <DmngNamePartKind Kind>
struct DmngRsltNamePartKSel {
  /// Type selected by kind selector (fall-back: base type).
  using Type = DmngRsltNamePart;
};

template <>
struct DmngRsltNamePartKSel<DNPK_Operator> {
  using Type = DmngRsltOpNamePart;
};

template <>
struct DmngRsltNamePartKSel<DNPK_Constructor> {
  using Type = DmngRsltCtorDtorNamePart;
};

template <>
struct DmngRsltNamePartKSel<DNPK_Destructor> {
  using Type = DmngRsltCtorDtorNamePart;
};

template <>
struct DmngRsltNamePartKSel<DNPK_Source> {
  using Type = DmngRsltSrcNamePart;
};

template <>
struct DmngRsltNamePartKSel<DNPK_UnnamedType> {
  using Type = DmngRsltUnmTypeNamePart;
};

template <>
struct DmngRsltNamePartKSel<DNPK_TemplateParam> {
  using Type = DmngRsltTParamNamePart;
};

template <>
struct DmngRsltNamePartKSel<DNPK_Decltype> {
  using Type = DmngRsltDecltypeNamePart;
};

template <>
struct DmngRsltNamePartKSel<DNPK_DataMember> {
  using Type = DmngRsltSrcNamePart;
};

template <DmngNamePartKind Kind>
using DmngRsltNamePartKSelT = typename DmngRsltNamePartKSel<Kind>::Type;


/// \brief Kind selector helper for derivates of DmngRsltType.
///
/// It allows to select derived type from DmngRsltType based on name kind.
///
/// \tparam Kind Kind of derived type which will be selected.
template <DmngTypeKind TypeKind>
struct DmngRsltTypeKSel {
  /// Type selected by kind selector (fall-back: base type).
  using Type = DmngRsltType;
};

template <>
struct DmngRsltTypeKSel<DTK_Builtin> {
  using Type = DmngRsltBuiltinType;
};

template <>
struct DmngRsltTypeKSel<DTK_Function> {
  using Type = DmngRsltFuncType;
};

template <>
struct DmngRsltTypeKSel<DTK_TypeName> {
  using Type = DmngRsltTypeNameType;
};

template <>
struct DmngRsltTypeKSel<DTK_Array> {
  using Type = DmngRsltArrayVecType;
};

template <>
struct DmngRsltTypeKSel<DTK_Vector> {
  using Type = DmngRsltArrayVecType;
};

template <>
struct DmngRsltTypeKSel<DTK_PointerToMember> {
  using Type = DmngRsltPtr2MmbrType;
};

template <>
struct DmngRsltTypeKSel<DTK_TemplateParam> {
  using Type = DmngRsltTParamType;
};

template <>
struct DmngRsltTypeKSel<DTK_Decltype> {
  using Type = DmngRsltDecltypeType;
};

template <>
struct DmngRsltTypeKSel<DTK_Pointer> {
  using Type = DmngRsltQualType;
};

template <>
struct DmngRsltTypeKSel<DTK_LValueRef> {
  using Type = DmngRsltQualType;
};

template <>
struct DmngRsltTypeKSel<DTK_RValueRef> {
  using Type = DmngRsltQualType;
};

template <>
struct DmngRsltTypeKSel<DTK_C2000Complex> {
  using Type = DmngRsltQualType;
};

template <>
struct DmngRsltTypeKSel<DTK_C2000Imaginary> {
  using Type = DmngRsltQualType;
};

template <>
struct DmngRsltTypeKSel<DTK_PackExpansion> {
  using Type = DmngRsltQualType;
};

template <>
struct DmngRsltTypeKSel<DTK_QualGroup> {
  using Type = DmngRsltQualGrpType;
};

template <DmngTypeKind TypeKind>
using DmngRsltTypeKSelT = typename DmngRsltTypeKSel<TypeKind>::Type;


/// \brief Kind selector helper for derivates of DmngRsltExpr.
///
/// It allows to select derived type from DmngRsltExpr based on name kind.
///
/// \tparam Kind Kind of derived type which will be selected.
template <DmngExprKind ExprKind>
struct DmngRsltExprKSel {
  /// Type selected by kind selector (fall-back: base type).
  using Type = DmngRsltExpr;
};

template <>
struct DmngRsltExprKSel<DXK_TemplateParam> {
  using Type = DmngRsltTParamExpr;
};

template <>
struct DmngRsltExprKSel<DXK_Decltype> {
  using Type = DmngRsltDecltypeExpr;
};

template <>
struct DmngRsltExprKSel<DXK_Primary> {
  using Type = DmngRsltPrimaryExpr;
};

template <DmngExprKind ExprKind>
using DmngRsltExprKSelT = typename DmngRsltExprKSel<ExprKind>::Type;


// -----------------------------------------------------------------------------
// BASE AND VALUE CLASSES FOR RESULT OBJECTS (COLLECTION WRAPPERS)
// -----------------------------------------------------------------------------

/// \brief Information about name parts in result (name parts collection).
///
/// This class is abstract.
class DmngRsltNamePartsBase {
public:
  /// Ordered collection type containing name parts.
  using NamePartsColT = std::vector<std::shared_ptr<const DmngRsltNamePart>>;

  /// \brief Gets last part of name.
  std::shared_ptr<const DmngRsltNamePart> getLastPart() const {
    return Parts.empty() ? nullptr : Parts.back();
  }

  /// \brief Gets name parts.
  const NamePartsColT &getParts() const {
    return Parts;
  }


  // Manipulators.

  /// \brief Gets last part of name.
  std::shared_ptr<DmngRsltNamePart> getModifiableLastPart() {
    // Static collection with pointers have really no good way to create
    // read-only interface (collections are not covariant).
    return Parts.empty()
      ? nullptr
      : std::const_pointer_cast<DmngRsltNamePart>(Parts.back());
  }

  /// \brief Adds name part (by copy).
  ///
  /// Function asserts and does nothing if name part is null.
  void addPart(const std::shared_ptr<const DmngRsltNamePart> &Part) {
    assert(Part != nullptr && "Name part must exist in order to be added.");
    if (Part != nullptr)
      Parts.push_back(Part);
  }

  /// \brief Adds name part (by move).
  ///
  /// Function asserts and does nothing if name part is null.
  void addPart(std::shared_ptr<const DmngRsltNamePart> &&Part) {
    assert(Part != nullptr && "Name part must exist in order to be added.");
    if (Part != nullptr)
      Parts.push_back(std::move(Part));
  }

  /// \brief Adds name part (bind pointer).
  ///
  /// Function asserts and does nothing if name part is null.
  void addPart(const DmngRsltNamePart *Part) {
    assert(Part != nullptr && "Name part must exist in order to be added.");
    if (Part != nullptr)
      Parts.push_back(std::shared_ptr<const DmngRsltNamePart>(Part));
  }

  /// \brief Sets name part (by copy).
  ///
  /// After calling this method name parts have only one part set to value
  /// passsed as parameter.
  ///
  /// Function asserts and does nothing if name part is null.
  void setPart(const std::shared_ptr<const DmngRsltNamePart> &Part) {
    assert(Part != nullptr && "Name part must exist in order to be added.");
    if (Part != nullptr) {
      Parts.clear();
      Parts.push_back(Part);
    }
  }

  /// \brief Sets name part (by move).
  ///
  /// After calling this method name parts have only one part set to value
  /// passsed as parameter.
  ///
  /// Function asserts and does nothing if name part is null.
  void setPart(std::shared_ptr<const DmngRsltNamePart> &&Part) {
    assert(Part != nullptr && "Name part must exist in order to be added.");
    if (Part != nullptr) {
      Parts.clear();
      Parts.push_back(std::move(Part));
    }
  }

  /// \brief Sets name part (bind pointer).
  ///
  /// After calling this method name parts have only one part set to value
  /// passsed as parameter.
  ///
  /// Function asserts and does nothing if name part is null.
  void setPart(const DmngRsltNamePart *Part) {
    assert(Part != nullptr && "Name part must exist in order to be added.");
    if (Part != nullptr) {
      Parts.clear();
      Parts.push_back(std::shared_ptr<const DmngRsltNamePart>(Part));
    }
  }

protected:
  /// \brief Creates instance with one name part (by copy).
  explicit DmngRsltNamePartsBase(
      const std::shared_ptr<const DmngRsltNamePart> &Part) {
    addPart(Part);
  }

  /// \brief Creates instance with one name part (by move).
  explicit DmngRsltNamePartsBase(
      std::shared_ptr<const DmngRsltNamePart> &&Part) {
    addPart(std::move(Part));
  }

  /// \brief Creates instance with one name part (bind pointer).
  explicit DmngRsltNamePartsBase(const DmngRsltNamePart *Part) {
    addPart(Part);
  }

  /// \brief Creates empty instance (for example string literal names).
  DmngRsltNamePartsBase() = default;


  /// \brief Creates instance that is a copy of other instance.
  ///
  /// \param Other    Other instance which will be copied.
  /// \param DeepCopy Indicates that name parts should be cloned as well.
  DmngRsltNamePartsBase(const DmngRsltNamePartsBase &Other, bool DeepCopy);

public:
  /// \brief Ensure proper destruction in some inheritance scenarios.
  virtual ~DmngRsltNamePartsBase() = default;


private:
  /// Name parts.
  NamePartsColT Parts;
};

/// \brief Information about types participating in function/closure signature
///        in result (bare function type/signature type collection).
///
/// This class is abstract.
class DmngRsltSignatureTypesBase {
public:
  /// Ordered collection type containing information about types.
  using TypesColT = std::vector<std::shared_ptr<const DmngRsltType>>;
  /// Iterator range for parameter types.
  using ParamTypesItRangeT = DmngRsltConstItRange<TypesColT::const_iterator>;


  /// \brief Gets ordered collection of types which describes function/closure
  ///        parameters and possibly return type.
  ///
  /// Collection is empty if current name/type is not function name/type.
  /// For function names/types, it contains possibly return type and parameter
  /// types information (return type at beginning, if exists).
  /// This collection will is non-empty for function names. If function has
  /// no parameters, it is treated as containg one parameter of void type.
  const TypesColT &getSignatureTypes() const {
    return SignatureTypes;
  }

  /// \brief Gets value indicating that current (function or closure) name/type
  ///        has return type encoded in signature.
  ///
  /// Default implementation always return false.
  virtual bool hasReturnType() const {
    return false;
  }

  /// \brief Gets information about return type of current (function or closure)
  ///        name/type.
  ///
  /// \return Information about return type, or nullptr if name is not function
  ///         name/type or return type was not encoded in function signature.
  std::shared_ptr<const DmngRsltType> getReturnType() const {
    if (hasReturnType())
      return SignatureTypes.front();
    return nullptr;
  }

  /// \brief Gets start iterator for parameter types in signature (ordered).
  TypesColT::const_iterator getParamTypeBegin() const {
    auto Begin = SignatureTypes.begin();
    return hasReturnType() ? ++Begin : Begin;
  }

  /// \brief Gets end iterator for parameter types in signature.
  TypesColT::const_iterator getParamTypeEnd() const {
    return SignatureTypes.end();
  }

  /// \brief Gets iterator range for parameter types (for for-each).
  ParamTypesItRangeT getParamTypes() const {
    return ParamTypesItRangeT(getParamTypeBegin(), getParamTypeEnd());
  }

  // Manipulators.

  /// \brief Adds signature type (by copy).
  ///
  /// Function asserts and does nothing if type is null.
  void addSignatureType(const std::shared_ptr<const DmngRsltType> &Type) {
    assert(Type != nullptr && "Type must exist in order to be added.");
    if (Type != nullptr)
      SignatureTypes.push_back(Type);
  }

  /// \brief Adds signature type (by move).
  ///
  /// Function asserts and does nothing if type is null.
  void addSignatureType(std::shared_ptr<const DmngRsltType> &&Type) {
    assert(Type != nullptr && "Type must exist in order to be added.");
    if (Type != nullptr)
      SignatureTypes.push_back(std::move(Type));
  }

  /// \brief Adds signature type (bind pointer).
  ///
  /// Function asserts and does nothing if type is null.
  void addSignatureType(const DmngRsltType *Type) {
    assert(Type != nullptr && "Type must exist in order to be added.");
    if (Type != nullptr)
      SignatureTypes.push_back(std::shared_ptr<const DmngRsltType>(Type));
  }

  /// \brief Resets signature.
  ///
  /// Removes all signature types.
  void resetSignature() {
    SignatureTypes.clear();
  }


protected:
  /// \brief Creates new instance of base class.
  DmngRsltSignatureTypesBase() = default;
public:
  /// \brief Ensure proper destruction in some inheritance scenarios.
  virtual ~DmngRsltSignatureTypesBase() = default;


private:
  /// Function signature (consist at least one element, i.e. void).
  TypesColT SignatureTypes;
};

/// \brief Template argument information.
class DmngRsltTArg {
public:
  /// Type of collection which contains information about template argument
  /// pack.
  using TArgsPackColT = std::vector<DmngRsltTArg>;


  /// \brief Indicates that template argument is an expression.
  bool isExpression() const;

  /// \brief Indicates that template argument is a type.
  bool isType() const;

  /// \brief Indicates that template argument is argument pack.
  bool isPack() const;


  /// \brief Gets value of template argument as expression.
  ///
  /// \return Shared pointer with value of argument, or empty pointer if
  ///         template argument is not expression.
  std::shared_ptr<const DmngRsltExpr> getExpression() const;

  /// \brief Gets value of template argument as type.
  ///
  /// \return Shared pointer with value of argument, or empty pointer if
  ///         template argument is not type.
  std::shared_ptr<const DmngRsltType> getType() const;

  /// \brief Gets value of template argument.
  ///
  /// \return Shared pointer with value of argument, or empty pointer if
  ///         template argument is argument pack.
  std::shared_ptr<const DmngRsltNode> getValue() const;

  /// \brief Gets template argument pack.
  ///
  /// \return Collection with argument pack, or empty pack if
  ///         template argument is not argument pack.
  const TArgsPackColT &getPack() const;


  /// \brief Gets empty template argument.
  static const DmngRsltTArg &getEmpty() {
    return EmptyArg;
  }

  // Manipulators.

  /// \brief Adds argument to current argument pack (by copy).
  void addPackArg(const DmngRsltTArg &Arg);

  /// \brief Adds argument to current argument pack (by move).
  void addPackArg(DmngRsltTArg &&Arg);


  /// \brief Creates new instance of template argument (expression).
  explicit DmngRsltTArg(const std::shared_ptr<const DmngRsltExpr> &Value);
  /// \brief Creates new instance of template argument (expression).
  explicit DmngRsltTArg(std::shared_ptr<const DmngRsltExpr> &&Value);

  /// \brief Creates new instance of template argument (type).
  explicit DmngRsltTArg(const std::shared_ptr<const DmngRsltType> &Value);
  /// \brief Creates new instance of template argument (type).
  explicit DmngRsltTArg(std::shared_ptr<const DmngRsltType> &&Value);

  /// \brief Creates new instance of template argument (pack).
  DmngRsltTArg() = default;


private:
  /// Value of current template argument.
  std::shared_ptr<const DmngRsltNode> Value;
  /// Template argument pack.
  TArgsPackColT Pack;

  /// Represents empty template parameter pack.
  static const TArgsPackColT EmpyPack;
  /// Represents empty template argument.
  static const DmngRsltTArg EmptyArg;
};

/// \brief Information about template arguments in result (template args
///        collection).
///
/// This class is abstract.
class DmngRsltTArgsBase {
public:
  /// Type of collection which contains information about template arguments.
  using TArgsColT = std::vector<DmngRsltTArg>;


  /// \brief Gets value indicating that current entity is template instance
  ///        or specialization.
  bool isTemplate() const {
    return !TemplateArgs.empty();
  }

  /// \brief Gets collection with information about template arguments
  ///        of current entity.
  const TArgsColT &getTemplateArgs() const {
    return TemplateArgs;
  }

  // Manipulators.

  /// \brief Adds information about template argument (by copy).
  ///
  /// Function asserts and does nothing if template argument is null.
  void addTemplateArg(const DmngRsltTArg &Arg) {
    TemplateArgs.push_back(Arg);
  }

  /// \brief Adds information about template argument (by move).
  ///
  /// Function asserts and does nothing if template argument is null.
  void addTemplateArg(DmngRsltTArg &&Arg) {
    TemplateArgs.push_back(std::move(Arg));
  }


protected:
  /// \brief Creates new instance of base class.
  DmngRsltTArgsBase() = default;
public:
  /// \brief Ensure proper destruction in some inheritance scenarios.
  virtual ~DmngRsltTArgsBase() = default;


private:
  /// Collection of template arguments. If it is empty, the current entity
  /// is not a template.
  TArgsColT TemplateArgs;
};

/// \brief Vendor-extended qualifier information.
class DmngRsltVendorQual : public DmngRsltTArgsBase {
public:
  /// \brief Gets vendor-extended qualfier name.
  const std::string &getName() const {
    return Name;
  }

  // Manipulators.

  /// \brief Creates new instance of qualifier with specific name.
  DmngRsltVendorQual(const std::string &Name)
    : Name(Name) {
    assert(!this->Name.empty() && "Qualifier name must exist.");
  }

  /// \brief Creates new instance of qualifier with specific name.
  DmngRsltVendorQual(std::string &&Name)
    : Name(std::move(Name)) {
    assert(!this->Name.empty() && "Qualifier name must exist.");
  }

  /// \brief Creates empty qualifier (no name).
  DmngRsltVendorQual() = default;

private:
  /// Vendor-extended qualifier name.
  std::string Name;
};

/// \brief Information about vendor-extended qualifiers and specifiers in result
///        (qualifiers collection).
///
/// This class is abstract.
class DmngRsltVendorQualsBase {
public:
  /// Type of collection which conatins information about vendor qualifiers.
  using VQualsT = std::vector<DmngRsltVendorQual>;


  /// \brief Gets value indicating that current entity has got vendor-extended
  ///        qualifiers.
  bool hasVendorQuals() const {
    return !Quals.empty() || AsQuals != DASQ_None;
  }

  /// \brief Gets collection with information about vendor-extended qualifiers
  ///        of current entity.
  const VQualsT& getVendorQuals() const {
    return Quals;
  }

  /// \brief Gets address space qualifiers.
  DmngAddressSpaceQuals getAsQuals() const {
    return AsQuals;
  }

  // Manipulators.

  /// \brief Adds information about vendor-extended qualifier (by copy).
  ///
  /// Function asserts and does nothing if qualifier is empty.
  void addVendorQual(const std::string &Qual) {
    assert(!Qual.empty() && "Qualifier must exist in order to be added.");
    if (!Qual.empty())
      Quals.push_back(Qual);
  }

  /// \brief Adds information about vendor-extended qualifier (by move).
  ///
  /// Function asserts and does nothing if qualifier is empty.
  void addVendorQual(std::string &&Qual) {
    assert(!Qual.empty() && "Qualifier must exist in order to be added.");
    if (!Qual.empty())
      Quals.push_back(std::move(Qual));
  }

  /// \brief Adds information about vendor-extended qualifier (by copy).
  void addVendorQual(const DmngRsltVendorQual &Qual) {
    if (!Qual.getName().empty())
      Quals.push_back(Qual);
  }

  /// \brief Adds information about vendor-extended qualifier (by move).
  void addVendorQual(DmngRsltVendorQual &&Qual) {
    if (!Qual.getName().empty())
      Quals.push_back(std::move(Qual));
  }

  /// \brief Sets address space qualifiers (__local, __global, etc.).
  void setAsQuals(DmngAddressSpaceQuals Quals) {
    AsQuals = Quals;
  }

  /// \brief Resets vendor qualifiers.
  void resetVendorQuals() {
    Quals.clear();
    AsQuals = DASQ_None;
  }

protected:
  /// \brief Creates new instance of base class.
  DmngRsltVendorQualsBase() : AsQuals(DASQ_None) {}
public:
  /// \brief Ensure proper destruction in some inheritance scenarios.
  virtual ~DmngRsltVendorQualsBase() = default;


private:
  /// Collection of vendor-extended qualifiers that do not have special
  /// parsing (i.e. address space has special treatment).
  VQualsT Quals;
  /// Address space qualifiers.
  DmngAddressSpaceQuals AsQuals;
};

/// \brief Adjustment offset (of "this"/result pointer) in demangled result.
///
/// The class is immutable. Use constructor to set to proper value.
// Nodes: <call-offset>
class DmngRsltAdjustOffset {
public:
  /// \brief Indicates that adjustment is zero.
  ///
  /// Returnts value indicating that current adjustment does not adjust
  /// anything. It is non-virtual with no base offset.
  bool isZero() const {
    return !IsVirtual && BaseOffset == 0;
  }

  /// \brief Indicates that adjustment is done via virtual base.
  ///
  /// Virtual adjustment uses two offset to adjust pointer to target base.
  /// First, non-virtual offset is used to find nearest virtual base of full
  /// object (that is non-virtual base from perspecive of current object).
  /// Second, is vcall offset in virtual base table that is used for final
  /// adjustment of pointer to target base.
  ///
  /// If it is true, both getBaseOffset() and getVCallOffset() are valid;
  /// otherwise, only getBaseOffset() is valid - getVCallOffset() returns 0.
  bool isVirtual() const {
    return IsVirtual;
  }

  /// \brief Non-virtual offset used to adjust pointer to base.
  ///
  /// If isVirtual() is true, the offset is used to locate nearest virtual
  /// base of full object that is non-virtual from point of current object.
  /// If isVirtual() is false, the offset is used directly to adjust to
  /// target base.
  long long getBaseOffset() const {
    return BaseOffset;
  }

  /// \brief Virtual (vcall) offset in virtual base table.
  ///
  /// vcall offset in virtual base that is used for final adjustment of pointer
  /// to target base.
  long long getVCallOffset() const {
    return IsVirtual ? VCallOffset : 0;
  }


  /// \brief Creates default instance (with no adjustment).
  DmngRsltAdjustOffset() : BaseOffset(0), VCallOffset(0), IsVirtual(false) {}

  /// \brief Creates instance with non-virtual adjustment.
  ///
  /// \param BaseOffset Offset to target base.
  explicit DmngRsltAdjustOffset(long long BaseOffset)
    : BaseOffset(BaseOffset), VCallOffset(0), IsVirtual(false) {}

  /// \brief Creates instance with virtual (base) adjustment.
  ///
  /// \param VBaseOffset Offset to nearest virtual base.
  /// \param VCallOffset vcall offset in virtual base table.
  DmngRsltAdjustOffset(long long VBaseOffset, long long VCallOffset)
    : BaseOffset(VBaseOffset), VCallOffset(VCallOffset), IsVirtual(true) {}


protected:
  /// Non-virtual offset for adjustment (either direct adjustment to target
  /// base or adjustment to nearest virtual base of full object that is
  /// non-virtual for current object).
  long long BaseOffset;
  /// Virtual offset (vcall offset in virtual base for final adjustment).
  long long VCallOffset;
  /// Indicates that adjustment is done via virtual base.
  bool IsVirtual;
};


// -----------------------------------------------------------------------------
// NODE CLASSES FOR RESULT OBJECTS
// -----------------------------------------------------------------------------

/// \brief Generic information node in result.
///
/// The class is intended for inheritance.
class DmngRsltNode : public std::enable_shared_from_this<DmngRsltNode> {
public:
  /// \brief Gets kind of current node.
  DmngNodeKind getNodeKind() const {
    return NodeKind;
  }

  /// \brief Gets current node information.
  ///
  /// \return Name information (as shared pointer).
  std::shared_ptr<const DmngRsltNode> getNode() const {
    return shared_from_this();
  }

  /// \brief Gets node specific information.
  ///
  /// \tparam Kind Kind of node for which information will be retrieved.
  /// \return Specific node information, or empty shared pointer
  ///         if there is no information connected to selected Kind.
  template <DmngNodeKind NodeKind>
  std::shared_ptr<const DmngRsltNodeKSelT<NodeKind>> getAs() const {
    using DerivedT = DmngRsltNodeKSelT<NodeKind>;

    if (NodeKind == this->NodeKind)
      return std::static_pointer_cast<const DerivedT>(getNode());
    return nullptr;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  virtual std::shared_ptr<DmngRsltNode> clone() const = 0;

  // Manipulators.

  /// \brief Gets current node information.
  ///
  /// \return Name information (as shared pointer).
  std::shared_ptr<DmngRsltNode> getNode() {
    return shared_from_this();
  }

  /// \brief Gets node specific information.
  ///
  /// \tparam Kind Kind of node for which information will be retrieved.
  /// \return Specific node information, or empty shared pointer
  ///         if there is no information connected to selected Kind.
  template <DmngNodeKind NodeKind>
  std::shared_ptr<DmngRsltNodeKSelT<NodeKind>> getAs() {
    using DerivedT = DmngRsltNodeKSelT<NodeKind>;

    if (NodeKind == this->NodeKind)
      return std::static_pointer_cast<DerivedT>(getNode());
    return nullptr;
  }


protected:
  /// \brief Creates (base) instance of information node with specific kind.
  explicit DmngRsltNode(DmngNodeKind NodeKind)
    : NodeKind(NodeKind) {}

public:
  /// \brief Ensure proper destruction.
  virtual ~DmngRsltNode() = default;


private:
  /// Kind of node.
  DmngNodeKind NodeKind;
};


/// \brief Information about expression.
class DmngRsltExpr : public DmngRsltNode {
public:
  /// \brief Get kind of expression information.
  DmngExprKind getKind() const {
    return ExprKind;
  }


  /// \brief Gets expression specific information.
  ///
  /// \tparam ExprKind Kind of expression for which information
  ///                  will be retrieved.
  /// \return Specific expression information, or empty shared pointer
  ///         if there is no information connected to selected Kind.
  template <DmngExprKind ExprKind>
  std::shared_ptr<const DmngRsltExprKSelT<ExprKind>> getAs() const {
    using DerivedT = DmngRsltExprKSelT<ExprKind>;

    if (ExprKind == this->ExprKind)
      return std::static_pointer_cast<const DerivedT>(getNode());
    return nullptr;
  }

  // Manipulators.

  /// \brief Gets expression specific information.
  ///
  /// \tparam ExprKind Kind of expression for which information
  ///                  will be retrieved.
  /// \return Specific expression information, or empty shared pointer
  ///         if there is no information connected to selected Kind.
  template <DmngExprKind ExprKind>
  std::shared_ptr<DmngRsltExprKSelT<ExprKind>> getAs() {
    using DerivedT = DmngRsltExprKSelT<ExprKind>;

    if (ExprKind == this->ExprKind)
      return std::static_pointer_cast<DerivedT>(getNode());
    return nullptr;
  }


protected:
  /// \brief Creates new instance of base for information about expressions.
  DmngRsltExpr(DmngExprKind Kind)
    : DmngRsltNode(DNDK_Expr), ExprKind(Kind) {}


private:
  /// Kind of demangled expression represented by current node.
  DmngExprKind ExprKind;
};

/// \brief Information about decltype() expression.
class DmngRsltDecltypeExpr : public DmngRsltExpr {
public:
  /// \brief Gets expression inside decltype().
  std::shared_ptr<const DmngRsltExpr> getExpression() const {
    return Expression;
  }

  /// \brief Indicates that decltype() is based on simple expression
  ///        (id expression or member access).
  bool isSimple() const {
    return IsSimple;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltDecltypeExpr>(*this);
  }

  // Manipulators.

  /// \brief Creates instance of information about decltype() expression
  ///        (by copy).
  explicit DmngRsltDecltypeExpr(
      const std::shared_ptr<const DmngRsltExpr> &Expression,
      bool IsSimple = false)
    : DmngRsltExpr(DXK_Decltype), Expression(Expression), IsSimple(IsSimple) {
    assert(this->Expression != nullptr &&
           "Expression in decltype() must exist.");
  }

  /// \brief Creates instance of information about decltype() expression
  ///        (by move).
  explicit DmngRsltDecltypeExpr(
      std::shared_ptr<const DmngRsltExpr> &&Expression, bool IsSimple = false)
    : DmngRsltExpr(DXK_Decltype), Expression(std::move(Expression)),
      IsSimple(IsSimple) {
    assert(this->Expression != nullptr &&
           "Expression in decltype() must exist.");
  }


private:
  /// Expression in decltype().
  std::shared_ptr<const DmngRsltExpr> Expression;
  /// Indicates that decltype() is base on simple expression (id expression or
  /// class member access).
  bool IsSimple;
};

/// \brief Information about template parameter expression.
class DmngRsltTParamExpr : public DmngRsltExpr {
public:
  /// \brief Gets index of referred template argument by current parameter.
  unsigned getReferredTArgIdx() const {
    return ReferredTArgIdx;
  }

  /// \brief Gets referred template argument by current parameter.
  ///
  /// \return Pointer to referred template argument, or empty shared pointer
  ///         if template parameter refers to invalid argument.
  std::shared_ptr<const DmngRsltTArg> getReferredTArg() const {
    return ReferredTArg;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltTParamExpr>(*this);
  }

  // Manipulators.

  /// \brief Sets referred template argument by current parameter (by copy).
  void setReferredTArg(const std::shared_ptr<const DmngRsltTArg> &Arg) {
    ReferredTArg = Arg;
  }

  /// \brief Sets referred template argument by current parameter (by move).
  void setReferredTArg(std::shared_ptr<const DmngRsltTArg> &&Arg) {
    ReferredTArg = Arg;
  }

  /// \brief Creates information about template parameter expression.
  ///
  /// \param TArgIdx Index of referred template argument by newly created
  ///                parameter.
  explicit DmngRsltTParamExpr(unsigned TArgIdx)
    : DmngRsltExpr(DXK_TemplateParam), ReferredTArgIdx(TArgIdx) {}


private:
  /// Index or referred template argument.
  unsigned ReferredTArgIdx;
  /// Resolved referred template argument.
  std::shared_ptr<const DmngRsltTArg> ReferredTArg;
};

/// \brief Information about primary expression (literals).
class DmngRsltPrimaryExpr : public DmngRsltExpr {
public:
  /// \brief Type of content that represents value of literal.
  enum LiteralContentType {
    Void, ///< No value/content is used (literal has only type or name).
    UInt, ///< Unsigned integer.
    SInt, ///< Signed integer.
    Bool, ///< Boolean.
    // NOTE: There is no need to handle floating-points here (for now).
    //Float,
    //Double,
    //FloatComplex,
    //DoubleComplex,
  };

  /// \brief Gets value indicating that primary expression is external name
  ///        (name of function/entity used as reference).
  bool isExternalName() const {
    return TypeOrName->getNodeKind() == DNDK_Name;
  }

  /// \brief Gets value indicating that primary expression is ordinary literal
  ///        with optional content and type.
  bool isLiteral() const {
    return TypeOrName->getNodeKind() == DNDK_Type;
  }

  /// \brief Gets external name identified by current primary expression
  ///        (name of function/entity used as reference).
  ///
  /// \return External name information (shared pointer), or empty pointer
  ///         if current primary expression does not refer to external name.
  std::shared_ptr<const DmngRsltName> getExternalName() const {
    return TypeOrName->getAs<DNDK_Name>();
  }

  /// \brief Gets type of literal in primary expression.
  ///
  /// \return Type information (shared pointer), or empty pointer
  ///         if current primary expression does not refer to ordinary literal.
  std::shared_ptr<const DmngRsltType> getLiteralType() const {
    return TypeOrName->getAs<DNDK_Type>();
  }

  /// \brief Gets content type of literal.
  LiteralContentType getContentType() const {
    return ContentType;
  }

  /// \brief Gets content as unsigned integer.
  ///
  /// \return Content of literal interpreted as unsigned integer.
  unsigned long long getContentAsUInt() const {
    return UIntValue;
  }

  /// \brief Gets content as signed integer.
  ///
  /// \return Content of literal interpreted as signed integer.
  long long getContentAsSInt() const {
    return SIntValue;
  }

  /// \brief Gets content as boolean.
  ///
  /// \return Content of literal interpreted as boolean.
  bool getContentAsBool() const {
    return UIntValue != 0;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltPrimaryExpr>(*this);
  }

  // Manipulators.

  /// \brief Creates instance of information about primary expression
  ///        which is external name (of function/entity).
  template <typename ExternalNameT,
            typename = SubnodeEnablerT<ExternalNameT, const DmngRsltName>>
  explicit DmngRsltPrimaryExpr(ExternalNameT &&ExternalName)
    : DmngRsltExpr(DXK_Primary),
      TypeOrName(std::forward<ExternalNameT>(ExternalName)),
      ContentType(Void) {
    assert(this->TypeOrName != nullptr &&
      "External name referred in primary expression must exist.");
  }

  /// \brief Creates instance of information about primary expression
  ///        which is ordinary literal (with no content).
  template <typename LiteralTypeT,
            typename = SubnodeEnablerT<LiteralTypeT, const DmngRsltType>,
            typename = void>
  explicit DmngRsltPrimaryExpr(LiteralTypeT &&LiteralType)
    : DmngRsltExpr(DXK_Primary),
      TypeOrName(std::forward<LiteralTypeT>(LiteralType)),
      ContentType(Void) {
    assert(this->TypeOrName != nullptr &&
           "Type of literal must exist.");
  }

  /// \brief Creates instance of information about primary expression
  ///        which is ordinary literal (with unsigned integer content).
  template <typename LiteralTypeT,
            typename = SubnodeEnablerT<LiteralTypeT, const DmngRsltType>>
  DmngRsltPrimaryExpr(LiteralTypeT &&LiteralType, unsigned long long Content)
    : DmngRsltExpr(DXK_Primary),
      TypeOrName(std::forward<LiteralTypeT>(LiteralType)),
      UIntValue(Content), ContentType(UInt) {
    assert(this->TypeOrName != nullptr &&
           "Type of literal must exist.");
  }

  /// \brief Creates instance of information about primary expression
  ///        which is ordinary literal (with signed integer content).
  template <typename LiteralTypeT,
            typename = SubnodeEnablerT<LiteralTypeT, const DmngRsltType>>
  DmngRsltPrimaryExpr(LiteralTypeT &&LiteralType, long long Content)
    : DmngRsltExpr(DXK_Primary),
      TypeOrName(std::forward<LiteralTypeT>(LiteralType)),
      SIntValue(Content), ContentType(SInt) {
    assert(this->TypeOrName != nullptr &&
           "Type of literal must exist.");
  }

  /// \brief Creates instance of information about primary expression
  ///        which is ordinary literal (with boolean content).
  template <typename LiteralTypeT,
            typename = SubnodeEnablerT<LiteralTypeT, const DmngRsltType>>
  DmngRsltPrimaryExpr(LiteralTypeT &&LiteralType, bool Content)
    : DmngRsltExpr(DXK_Primary),
      TypeOrName(std::forward<LiteralTypeT>(LiteralType)),
      UIntValue(Content), ContentType(Bool) {
    assert(this->TypeOrName != nullptr &&
           "Type of literal must exist.");
  }


private:
  /// Type of literal value or name that names literal entity in current
  /// primary expression.
  std::shared_ptr<const DmngRsltNode> TypeOrName;
  union {
    unsigned long long UIntValue; ///< Content as unsigned integer.
    long long SIntValue;          ///< Content as signed integer.
    // NOTE: There is no need to handle floating-points here (for now).
    //float SFValue;
    //double DFValue;
    //float SCValue[2];
    //double DCValue[2];
  };
  /// Type of content for literal.
  LiteralContentType ContentType;
};


/// \brief Information about type.
class DmngRsltType : public DmngRsltNode {
public:
  /// \brief Get kind of type information.
  DmngTypeKind getKind() const {
    return TypeKind;
  }

  /// \brief Gets type specific information.
  ///
  /// \tparam TypeKind Kind of type for which information will be retrieved.
  /// \return Specific type information, or empty shared pointer
  ///         if there is no information connected to selected Kind.
  template <DmngTypeKind TypeKind>
  std::shared_ptr<const DmngRsltTypeKSelT<TypeKind>> getAs() const {
    using DerivedT = DmngRsltTypeKSelT<TypeKind>;

    if (TypeKind == this->TypeKind)
      return std::static_pointer_cast<const DerivedT>(getNode());
    return nullptr;
  }

  // Manipulators.

  /// \brief Gets type specific information.
  ///
  /// \tparam TypeKind Kind of type for which information will be retrieved.
  /// \return Specific type information, or empty shared pointer
  ///         if there is no information connected to selected Kind.
  template <DmngTypeKind TypeKind>
  std::shared_ptr<DmngRsltTypeKSelT<TypeKind>> getAs() {
    using DerivedT = DmngRsltTypeKSelT<TypeKind>;

    if (TypeKind == this->TypeKind)
      return std::static_pointer_cast<DerivedT>(getNode());
    return nullptr;
  }


protected:
  /// \brief Creates new instance of base for information about types.
  explicit DmngRsltType(DmngTypeKind TypeKind)
    : DmngRsltNode(DNDK_Type), TypeKind(TypeKind) {}


private:
  /// Kind of demangled type represented by current node.
  DmngTypeKind TypeKind;
};

/// \brief Information about type which represents built-in type.
class DmngRsltBuiltinType : public DmngRsltType {
public:
  /// \brief Gets numeric identifier of built-in type.
  DmngBuiltinType getBuiltinType() const {
    return Type;
  }

  /// \brief Indicates that built-in is vendor-extended built-in type.
  bool isVendorBuiltinType() const {
    return Type == DBT_Vendor;
  }

  /// \brief Gets name of vendor-extended built-in.
  ///
  /// \return Name of built-in, or empty string if built-in is not
  ///         vendor-extended.
  const std::string &getVendorName() const {
    return isVendorBuiltinType() ? VendorName : DmngRsltEmptyString;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltBuiltinType>(*this);
  }

  // Manipulators.

  /// \brief Creates (non-vendor-extended) built-in type.
  explicit DmngRsltBuiltinType(DmngBuiltinType BuiltinType)
    : DmngRsltType(DTK_Builtin), Type(BuiltinType) {
    assert(!isVendorBuiltinType() && "Vendor built-in type requires name.");
  }

  /// \brief Creates vendor-extended built-in type (by copy).
  explicit DmngRsltBuiltinType(const std::string &VendorName)
    : DmngRsltType(DTK_Builtin), Type(DBT_Vendor), VendorName(VendorName) {
    assert(!VendorName.empty() && "Vendor built-in type requires name.");
  }

  /// \brief Creates vendor-extended built-in type (by move).
  explicit DmngRsltBuiltinType(std::string &&VendorName)
    : DmngRsltType(DTK_Builtin), Type(DBT_Vendor),
      VendorName(std::move(VendorName)) {
    assert(!this->VendorName.empty() && "Vendor built-in type requires name.");
  }


private:
  /// Identification of built-in type.
  DmngBuiltinType Type;
  /// For vendor-specific built-in types, name of the type.
  std::string VendorName;
};

/// \brief Information about type which represents function type.
class DmngRsltFuncType : public DmngRsltType,
                         public DmngRsltSignatureTypesBase,
                         public DmngRsltVendorQualsBase {
public:
  /// \brief Gets cvr-qualifiers for current (member) function type.
  DmngCvrQuals getCvrQuals() const {
    return CvrQuals;
  }

  /// \brief Gets reference qualifier of current (member) function type.
  DmngRefQuals getRefQuals() const {
    return RefQuals;
  }

  /// \brief Indicates that function type was marked with: extern "C".
  bool isExternC() const {
    return IsExternC;
  }

  /// \brief Indicates that function type has return type in signature.
  bool hasReturnType() const override
  {
    // Function types have return type except rare encoding: FvE (usually
    // instead of FvvE) for: void ().
    return getSignatureTypes().size() > 1;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltFuncType>(*this);
  }

  // Manipulators.

  /// \brief Sets cvr-qualifiers (const, volatile, restrict, etc.) for
  ///        current (member) function type.
  void setCvrQuals(DmngCvrQuals Quals) {
    CvrQuals = Quals;
  }

  /// \brief Sets reference qualifiers (&, &&, etc.) for current (member)
  ///        function type.
  void setRefQuals(DmngRefQuals Quals) {
    RefQuals = Quals;
  }

  /// \brief Marks function type with: extern "C".
  void setIsExternC(bool IsExternC = true) {
    this->IsExternC = IsExternC;
  }


  /// \brief Creates function type with initial single type (by copy).
  explicit DmngRsltFuncType(const std::shared_ptr<const DmngRsltType> &Type)
    : DmngRsltType(DTK_Function), CvrQuals(DCVQ_None), RefQuals(DRQ_None),
      IsExternC(false) {
    addSignatureType(Type);
  }

  /// \brief Creates function type with initial single type (by move).
  explicit DmngRsltFuncType(std::shared_ptr<const DmngRsltType> &&Type)
    : DmngRsltType(DTK_Function), CvrQuals(DCVQ_None), RefQuals(DRQ_None),
      IsExternC(false) {
    addSignatureType(std::move(Type));
  }

  /// \brief Creates function type with initial single type (bind pointer).
  explicit DmngRsltFuncType(const DmngRsltType *Type)
    : DmngRsltType(DTK_Function), CvrQuals(DCVQ_None), RefQuals(DRQ_None),
      IsExternC(false) {
    addSignatureType(Type);
  }


private:
  /// \brief cvr-qualifiers of function (not separable).
  DmngCvrQuals CvrQuals;
  /// \brief Reference qualifier of function (not separable).
  DmngRefQuals RefQuals;
  /// \brief Indicates that function is marked as: extern "C".
  bool IsExternC;
};

/// \brief Information about type which represents named or unnamed type.
class DmngRsltTypeNameType : public DmngRsltType {
public:
  /// \brief Gets optional elaboration of type name.
  DmngTypeNameKind getElaboration() const {
    return Elaboration;
  }

  /// \brief Get name node that describes current named or unnamed type.
  std::shared_ptr<const DmngRsltOrdinaryName> getTypeName() const {
    return TypeName;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltTypeNameType>(*this);
  }

  // Manipulators.

  /// \brief Creates type which represents type name (by copy).
  explicit DmngRsltTypeNameType(
      const std::shared_ptr<const DmngRsltOrdinaryName> &Name,
      DmngTypeNameKind Elaboration = DTNK_None)
    : DmngRsltType(DTK_TypeName), TypeName(Name), Elaboration(Elaboration) {
    assert(this->TypeName != nullptr &&
           "Type name refered by type node must exist.");
  }

  /// \brief Creates type which represents type name (by move).
  explicit DmngRsltTypeNameType(
      std::shared_ptr<const DmngRsltOrdinaryName> &&Name,
      DmngTypeNameKind Elaboration = DTNK_None)
    : DmngRsltType(DTK_TypeName), TypeName(std::move(Name)),
      Elaboration(Elaboration) {
    assert(this->TypeName != nullptr &&
           "Type name refered by type node must exist.");
  }


private:
  /// Name of the type (class, struct, union, closure, enum names).
  std::shared_ptr<const DmngRsltOrdinaryName> TypeName;
  /// Type of elaboration keyword used.
  DmngTypeNameKind Elaboration;
};

/// \brief Information about type which represents array or vector type.
class DmngRsltArrayVecType : public DmngRsltType {
public:
  /// \brief Gets element type of array or vector.
  std::shared_ptr<const DmngRsltType> getElemType() const {
    return ElemType;
  }

  /// \brief Indicates that array / vector has specified fixed size
  ///        (direct size of compile-time expression).
  ///
  /// It allows to differentiate 0-sized arrays from VLA, etc.
  bool isSizeSpecified() const {
    return IsSizeSpecified;
  }

  /// \brief Gets size of array or vector in elements.
  ///
  /// \return Size of collection type in elements, or 0 if size is unknown,
  ///         unspecified, run-time variable-length or specified by expression.
  unsigned long long getSize() const {
    return IsSizeSpecified && SizeExpr == nullptr ? Size : 0;
  }

  /// \brief Gets expression that describes size of array / vector.
  ///
  /// \return Expression, or empty shared pointer if size is unspecified,
  ///         unknown or VLA.
  std::shared_ptr<const DmngRsltExpr> getSizeExpr() const {
    return IsSizeSpecified ? SizeExpr : nullptr;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltArrayVecType>(*this);
  }

  // Manipulators.

private:
  /// \brief Creates new instance (unified constructor).
  template <typename ElemTypeT, typename SizeExprT>
  DmngRsltArrayVecType(bool IsVector, ElemTypeT &&ElemType,
                       bool IsSizeSpecified, unsigned long long Size,
                       SizeExprT &&SizeExpr)
    : DmngRsltType(IsVector ? DTK_Vector : DTK_Array),
      ElemType(std::forward<ElemTypeT>(ElemType)), Size(Size),
      SizeExpr(std::forward<SizeExprT>(SizeExpr)),
      IsSizeSpecified(IsSizeSpecified) {
    assert(this->ElemType != nullptr && "Element type must exist.");
  }

public:
  /// \brief Creates new instance of vector/array type with unspecified
  ///        size (for VLA, etc.).
  template <typename ElemTypeT,
            typename = SubnodeEnablerT<ElemTypeT, const DmngRsltType>>
  explicit DmngRsltArrayVecType(ElemTypeT &&ElemType, bool IsVector = false)
    : DmngRsltArrayVecType(IsVector, std::forward<ElemTypeT>(ElemType),
                           false, 0, nullptr) {}

  /// \brief Creates new instance of vector/array type with fixed size
  ///        (size as direct number).
  template <typename ElemTypeT,
            typename = SubnodeEnablerT<ElemTypeT, const DmngRsltType>>
  DmngRsltArrayVecType(ElemTypeT &&ElemType, unsigned long long Size,
                       bool IsVector = false)
    : DmngRsltArrayVecType(IsVector, std::forward<ElemTypeT>(ElemType),
                           true, Size, nullptr) {}

  /// \brief Creates new instance of vector/array type with fixed size
  ///        (size as compile-time instance-dependent expression).
  template <typename ElemTypeT, typename SizeExprT,
            typename = SubnodeEnablerT<ElemTypeT, const DmngRsltType>,
            typename = SubnodeEnablerT<SizeExprT, const DmngRsltExpr>>
  DmngRsltArrayVecType(ElemTypeT &&ElemType, SizeExprT &&SizeExpr,
                       bool IsVector = false)
    : DmngRsltArrayVecType(IsVector, std::forward<ElemTypeT>(ElemType),
                           true, 0, std::forward<SizeExprT>(SizeExpr)) {}


private:
  /// Type of element in array or vector.
  std::shared_ptr<const DmngRsltType> ElemType;
  /// Size of array or vector in elements (if fixed, not-dependent size).
  unsigned long long Size;
  std::shared_ptr<const DmngRsltExpr> SizeExpr;
  /// Indicates whether size and size expression is specified.
  /// false for array of undetermined or variable-length (VLA).
  bool IsSizeSpecified;
};

/// \brief Information about type which represents pointer to member.
class DmngRsltPtr2MmbrType : public DmngRsltType {
public:
  /// \brief Gets class type of current pointer to member.
  std::shared_ptr<const DmngRsltType> getClassType() const {
    return ClassType;
  }

  /// \brief Gets member type of current pointer to member.
  std::shared_ptr<const DmngRsltType> getMemberType() const {
    return MemberType;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltPtr2MmbrType>(*this);
  }

  // Manipulators.

  /// \brief Creates new instance of information about pointer to member type.
  template <typename ClassTypeT, typename MemberTypeT,
            typename = SubnodeEnablerT<ClassTypeT, const DmngRsltType>,
            typename = SubnodeEnablerT<MemberTypeT, const DmngRsltType>>
  DmngRsltPtr2MmbrType(ClassTypeT &&ClassType, MemberTypeT &&MemberType)
    : DmngRsltType(DTK_PointerToMember),
      ClassType(std::forward<ClassTypeT>(ClassType)),
      MemberType(std::forward<MemberTypeT>(MemberType)) {
    assert(this->ClassType != nullptr &&
           "Class type of pointer to member must exist.");
    assert(this->MemberType != nullptr &&
           "Member type of pointer to member must exist.");
  }

private:
  /// Class type.
  std::shared_ptr<const DmngRsltType> ClassType;
  /// Type of member (member in class specified by class type).
  std::shared_ptr<const DmngRsltType> MemberType;
};

/// \brief Information about type which represents template parameter or
///        template template parameter.
class DmngRsltTParamType : public DmngRsltType, public DmngRsltTArgsBase {
public:
  /// \brief Gets information about template parameter used as current
  ///        type.
  std::shared_ptr<const DmngRsltTParamExpr> getTemplateParam() const {
    return TemplateParam;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltTParamType>(*this);
  }

  // Manipulators.

  /// \brief Sets template parameter information (by copy).
  void setTemplateParam(
      const std::shared_ptr<const DmngRsltTParamExpr> &Param) {
    assert(Param != nullptr && "Template parameter must be defined.");
    TemplateParam = Param;
  }

  /// \brief Sets template parameter information (by move).
  void setTemplateParam(std::shared_ptr<const DmngRsltTParamExpr> &&Param) {
    assert(Param != nullptr && "Template parameter must be defined.");
    TemplateParam = std::move(Param);
  }

  /// \brief Sets template parameter information (bind pointer).
  void setTemplateParam(const DmngRsltTParamExpr *Param) {
    assert(Param != nullptr && "Template parameter must be defined.");
    TemplateParam = std::shared_ptr<const DmngRsltTParamExpr>(Param);
  }

  /// \brief Creates instance of information about type with template or
  ///        template template parameter info (by copy).
  explicit DmngRsltTParamType(
      const std::shared_ptr<const DmngRsltTParamExpr> &Param)
    : DmngRsltType(DTK_TemplateParam) {
    setTemplateParam(Param);
  }

  /// \brief Creates instance of information about type with template or
  ///        template template parameter info (by move).
  explicit DmngRsltTParamType(
      std::shared_ptr<const DmngRsltTParamExpr> &&Param)
    : DmngRsltType(DTK_TemplateParam) {
    setTemplateParam(std::move(Param));
  }

  /// \brief Creates instance of information about type with template or
  ///        template template parameter info (bind pointer).
  explicit DmngRsltTParamType(const DmngRsltTParamExpr *Param)
    : DmngRsltType(DTK_TemplateParam) {
    setTemplateParam(Param);
  }


private:
  /// Template parameter used as a type (its referred type is used).
  std::shared_ptr<const DmngRsltTParamExpr> TemplateParam;
};

/// \brief Information about type which represents (result of) decltype()
///        expression.
class DmngRsltDecltypeType : public DmngRsltType {
public:
  /// \brief Gets information about decltype() expression used as current
  ///        type information.
  std::shared_ptr<const DmngRsltDecltypeExpr> getDecltype() const {
    return Decltype;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltDecltypeType>(*this);
  }

  // Manipulators.

  /// \brief Creates instance with decltype() expression info (by copy).
  explicit DmngRsltDecltypeType(
      const std::shared_ptr<const DmngRsltDecltypeExpr> &Decltype)
    : DmngRsltType(DTK_Decltype), Decltype(Decltype) {
    assert(this->Decltype != nullptr && "Decltype expression must be defined.");
  }

  /// \brief Creates instance with decltype() expression info (by move).
  explicit DmngRsltDecltypeType(
      std::shared_ptr<const DmngRsltDecltypeExpr> &&Decltype)
    : DmngRsltType(DTK_Decltype), Decltype(std::move(Decltype)) {
    assert(this->Decltype != nullptr && "Decltype expression must be defined.");
  }

  /// \brief Creates instance with decltype() expression info (bind pointer).
  explicit DmngRsltDecltypeType(const DmngRsltDecltypeExpr *Decltype)
    : DmngRsltType(DTK_Decltype), Decltype(Decltype) {
    assert(this->Decltype != nullptr && "Decltype expression must be defined.");
  }


private:
  /// decltype() expression used as a type (its result type is used).
  std::shared_ptr<const DmngRsltDecltypeExpr> Decltype;
};

/// \brief Information about type which represents type qualifed by single
///        qualifier (usually order-sensitive qualifier).
class DmngRsltQualType : public DmngRsltType {
public:
  /// \brief Gets inner type qualified by current qualification.
  std::shared_ptr<const DmngRsltType> getInnerType() const {
    return InnerType;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltQualType>(*this);
  }

  // Manipulators.

private:
  /// \brief Creates new instance of qualified type (information).
  ///
  /// \param Type Inner type which will be qualified.
  template <typename InnerTypeT>
  DmngRsltQualType(DmngTypeKind TypeKind, InnerTypeT &&Type)
    : DmngRsltType(TypeKind), InnerType(std::forward<InnerTypeT>(Type)) {
    assert(this->InnerType != nullptr && "Inner type must be defined.");
  }

public:
  /// \brief Creates pointer qualified type (T -> T*).
  template <typename InnerTypeT>
  static DmngRsltQualType createPointer(InnerTypeT &&Type) {
    return DmngRsltQualType(DTK_Pointer, std::forward<InnerTypeT>(Type));
  }

  /// \brief Creates lvalue reference qualified type (T -> T&).
  template <typename InnerTypeT>
  static DmngRsltQualType createLValueRef(InnerTypeT &&Type) {
    return DmngRsltQualType(DTK_LValueRef, std::forward<InnerTypeT>(Type));
  }

  /// \brief Creates lvalue reference qualified type (T -> T&&).
  template <typename InnerTypeT>
  static DmngRsltQualType createRValueRef(InnerTypeT &&Type) {
    return DmngRsltQualType(DTK_RValueRef, std::forward<InnerTypeT>(Type));
  }

  /// \brief Creates 'complex' qualified type (T -> complex T).
  template <typename InnerTypeT>
  static DmngRsltQualType createComplex(InnerTypeT &&Type) {
    return DmngRsltQualType(DTK_C2000Complex, std::forward<InnerTypeT>(Type));
  }

  /// \brief Creates 'imaginary' qualified type (T -> imaginary T).
  template <typename InnerTypeT>
  static DmngRsltQualType createImaginary(InnerTypeT &&Type) {
    return DmngRsltQualType(DTK_C2000Imaginary, std::forward<InnerTypeT>(Type));
  }

  /// \brief Creates pack expansion of type (T -> T ...).
  template <typename InnerTypeT>
  static DmngRsltQualType createPackExpansion(InnerTypeT &&Type) {
    return DmngRsltQualType(DTK_PackExpansion, std::forward<InnerTypeT>(Type));
  }


private:
  /// Inner type which is qualified.
  std::shared_ptr<const DmngRsltType> InnerType;
};

/// \brief Information about type which represents type qualifed by group of
///        qualifier (usually order-insensitive qualifiers).
class DmngRsltQualGrpType : public DmngRsltType,
                            public DmngRsltVendorQualsBase {
public:
  /// \brief Gets inner type qualified by current qualification.
  std::shared_ptr<const DmngRsltType> getInnerType() const {
    return InnerType;
  }

  /// \brief Gets cvr-qualifiers for current name (usually used with member
  ///        functions).
  DmngCvrQuals getCvrQuals() const {
    return CvrQuals;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltQualGrpType>(*this);
  }

  // Manipulators.

  /// \brief Sets cvr-qualifiers (const, volatile, restrict, etc.).
  void setCvrQuals(DmngCvrQuals Quals) {
    CvrQuals = Quals;
  }

  /// \brief Creates new instance of qualified type (information).
  ///
  /// \param Type Inner type which will be qualified.
  template <typename InnerTypeT,
            typename = SubnodeEnablerT<InnerTypeT, const DmngRsltType>>
  explicit DmngRsltQualGrpType(InnerTypeT &&Type,
                               DmngCvrQuals Quals = DCVQ_None)
    : DmngRsltType(DTK_QualGroup), InnerType(std::forward<InnerTypeT>(Type)),
      CvrQuals(Quals) {
    assert(this->InnerType != nullptr && "Inner type must be defined.");
  }


private:
  /// Inner type which is qualified (by set of qualifiers).
  std::shared_ptr<const DmngRsltType> InnerType;
  /// \brief cvr-qualifiers of inner type.
  DmngCvrQuals CvrQuals;
};


/// \brief Information about name part in result.
// Nodes: <prefix>, <template-prefix>, <unqualified-name>
class DmngRsltNamePart : public DmngRsltNode, public DmngRsltTArgsBase {
public:
  /// \brief Gets kind of name part.
  DmngNamePartKind getPartKind() const {
    return PartKind;
  }

  /// \brief Indicates that current part describes data member.
  bool isDataMember() const {
    return PartKind == DNPK_DataMember;
  }

  /// \brief Gets name part specific information.
  ///
  /// \tparam Kind Kind of name part for which information will be retrieved.
  /// \return Specific name part information, or empty shared pointer
  ///         if there is no information connected to selected Kind.
  template <DmngNamePartKind PartKind>
  std::shared_ptr<const DmngRsltNamePartKSelT<PartKind>> getAs() const {
    using DerivedT = DmngRsltNamePartKSelT<PartKind>;

    if (PartKind == this->PartKind)
      return std::static_pointer_cast<const DerivedT>(getNode());
    return nullptr;
  }

  // Manipulators.

  /// \brief Gets name part specific information.
  ///
  /// \tparam Kind Kind of name part for which information will be retrieved.
  /// \return Specific name part information, or empty shared pointer
  ///         if there is no information connected to selected Kind.
  template <DmngNamePartKind PartKind>
  std::shared_ptr<DmngRsltNamePartKSelT<PartKind>> getAs() {
    using DerivedT = DmngRsltNamePartKSelT<PartKind>;

    if (PartKind == this->PartKind)
      return std::static_pointer_cast<DerivedT>(getNode());
    return nullptr;
  }


protected:
  /// \brief Creates instance of base class with specific kind of name part.
  explicit DmngRsltNamePart(DmngNamePartKind PartKind)
    : DmngRsltNode(DNDK_NamePart), PartKind(PartKind) {}


private:
  /// Kind of name part.
  DmngNamePartKind PartKind;
};

/// \brief Name part that describes operator (function).
class DmngRsltOpNamePart : public DmngRsltNamePart {
public:
  /// \brief Gets enum value that identifies current operator.
  DmngOperatorName getNameCode() const {
    return NameCode;
  }


  /// \brief Gets value indicating that current operator is conversion operator.
  bool isConversionOperator() const {
    return NameCode == DON_Convert;
  }

  /// \brief Gets destination type of conversion operator.
  ///
  /// \return Type which convert operator converts to, or empty shared pointer
  ///         if current operator is not conversion operator.
  std::shared_ptr<const DmngRsltType> getConvertTargetType() const {
    return isConversionOperator() ? ConvType : nullptr;
  }

  /// \brief Gets value indicating that current operator is literal operator.
  bool isLiteralOperator() const {
    return NameCode == DON_Literal;
  }

  /// \brief Gets suffix used by literal operator.
  ///
  /// \return Literal operator suffix, or empty string if current operator
  ///         is not literal operator.
  const std::string &getLiteralOperatorSuffix() const {
    return isLiteralOperator() ? ExtName : DmngRsltEmptyString;
  }

  /// \brief Gets value indicating that current operator is vendor-specific
  ///        operator.
  bool isVendorOperator() const {
    return NameCode == DON_Vendor;
  }

  /// \brief Gets name of vendor-specific operator.
  ///
  /// \return Name for custom operator, or empty string if current operator
  ///         is not vendor-specific.
  const std::string &getVendorOperatorName() const {
    return isVendorOperator() ? ExtName : DmngRsltEmptyString;
  }

  /// \brief Gets arity of vendor-specific operator (as stated in mangling).
  ///
  /// \return Arity from mangling, or 0 if current operator is not
  ///         vendor-specific.
  int getVendorOperatorArity() const {
    return isVendorOperator() ? Arity : 0;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltOpNamePart>(*this);
  }

  // Manipulators.

  /// \brief Creates operator (non-conversion/-literal/-vendor case).
  explicit DmngRsltOpNamePart(DmngOperatorName NameCode)
    : DmngRsltNamePart(DNPK_Operator), NameCode(NameCode), Arity(0) {
    assert(!isConversionOperator() && !isLiteralOperator() &&
           !isVendorOperator() &&
           "Conversion/literal/vendor operator is created with another ctors.");
  }

  /// \brief Creates conversion operator.
  explicit DmngRsltOpNamePart(
      const std::shared_ptr<const DmngRsltType> &ConvertTargetType)
    : DmngRsltNamePart(DNPK_Operator), NameCode(DON_Convert),
    ConvType(ConvertTargetType), Arity(0) {
    assert(ConvType != nullptr &&
           "Type to convert to must be defined.");
  }

  /// \brief Creates literal operator.
  explicit DmngRsltOpNamePart(const std::string &LiteralSuffix)
    : DmngRsltNamePart(DNPK_Operator), NameCode(DON_Literal),
      ExtName(LiteralSuffix), Arity(0) {
    assert(!ExtName.empty() && "Literal suffix must be specified.");
  }

  /// \brief Creates vendor-specific operator.
  DmngRsltOpNamePart(const std::string &VendorOperatorName, int Arity)
    : DmngRsltNamePart(DNPK_Operator), NameCode(DON_Vendor),
      ExtName(VendorOperatorName), Arity(Arity > 0 ? Arity : 1) {
    assert(this->Arity > 0 && "Vendor operator arity must be at least one.");
    assert(!ExtName.empty() && "Operator name must be specified.");
  }

private:
  /// Code which identifies operator name.
  DmngOperatorName NameCode;
  /// Extended name for operator (either vendor operator name or suffix for
  /// literal operator).
  std::string ExtName;
  /// Target type for conversion operator.
  std::shared_ptr<const DmngRsltType> ConvType;
  /// Arity (1 - unary, 2 - binary, etc.) of vendor operator.
  int Arity;
};

/// \brief Name part that describes constructor / destructor (special function).
class DmngRsltCtorDtorNamePart : public DmngRsltNamePart {
public:
  /// \brief Gets type of constructor or destructor.
  DmngCtorDtorType getType() const {
    return Type;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltCtorDtorNamePart>(*this);
  }

  // Manipulators.

  /// \brief Creates constructor or destructor information.
  DmngRsltCtorDtorNamePart(bool IsConstructor, DmngCtorDtorType Type)
    : DmngRsltNamePart(IsConstructor ? DNPK_Constructor : DNPK_Destructor),
      Type(Type) {}


private:
  /// Constructor/Destructor type.
  DmngCtorDtorType Type;
};

/// \brief Name part that describes in-source names (namespace names,
///        class names, data member names, etc.).
class DmngRsltSrcNamePart : public DmngRsltNamePart {
public:
  /// \brief Gets in-source name (name as stated in source).
  const std::string& getSourceName() const {
    return SourceName;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltSrcNamePart>(*this);
  }

  // Manipulators.

  /// \brief Creates name part which contains in-source name.
  explicit DmngRsltSrcNamePart(const std::string &SourceName,
                               bool IsDataMember = false)
    : DmngRsltNamePart(IsDataMember ? DNPK_DataMember : DNPK_Source),
      SourceName(SourceName) {
    assert(!this->SourceName.empty() &&
           "Name (from source code) must be specified.");
  }

private:
  /// Name of the part (the same as in source).
  std::string SourceName;
};

/// \brief Name part that represents unnamed type names (including closures).
class DmngRsltUnmTypeNamePart : public DmngRsltNamePart,
                                public DmngRsltSignatureTypesBase {
public:
  /// \brief Gets discriminating identifier which allows to differentiate
  ///        between two unnamed types.
  ///
  /// The identifier is unique for all unnamed types in specific scope. In
  /// namespace or global scope the identifier ordering is unspecified as long
  /// it does not cause collisions between unnamed types (even in different
  /// translation units). In local/class/member/parameter scope it is usually
  /// applied in lexical order.
  unsigned long long getId() const {
    return Id;
  }

  /// \brief Indicates that unnamed type is closure type.
  bool isClosure() const {
    return !getSignatureTypes().empty();
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltUnmTypeNamePart>(*this);
  }

  // Manipulators.

  /// Creates unnamed type (either normal or closure type).
  explicit DmngRsltUnmTypeNamePart(unsigned long long Id = 0)
    : DmngRsltNamePart(DNPK_UnnamedType), Id(Id) {}

private:
  /// Discriminator.
  unsigned long long Id;
};

/// \brief Name part that represents (unresolved) template parameter.
///
/// Template parameter refers to template argument. The argument in some
/// cases can be defined after template parameter.
class DmngRsltTParamNamePart : public DmngRsltNamePart {
public:
  /// \brief Gets information about template parameter used as current
  ///        name part.
  std::shared_ptr<const DmngRsltTParamExpr> getTemplateParam() const {
    return TemplateParam;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltTParamNamePart>(*this);
  }

  // Manipulators.

  /// \brief Sets template parameter information (by copy).
  void setTemplateParam(
      const std::shared_ptr<const DmngRsltTParamExpr> &Param) {
    assert(Param != nullptr && "Template parameter must be defined.");
    TemplateParam = Param;
  }

  /// \brief Sets template parameter information (by move).
  void setTemplateParam(std::shared_ptr<const DmngRsltTParamExpr> &&Param) {
    assert(Param != nullptr && "Template parameter must be defined.");
    TemplateParam = std::move(Param);
  }

  /// \brief Sets template parameter information (bind pointer).
  void setTemplateParam(const DmngRsltTParamExpr *Param) {
    assert(Param != nullptr && "Template parameter must be defined.");
    TemplateParam = std::shared_ptr<const DmngRsltTParamExpr>(Param);
  }

  /// \brief Creates instance with template parameter info (by copy).
  explicit DmngRsltTParamNamePart(
      const std::shared_ptr<const DmngRsltTParamExpr> &Param)
    : DmngRsltNamePart(DNPK_TemplateParam) {
    setTemplateParam(Param);
  }

  /// \brief Creates instance with template parameter info (by move).
  explicit DmngRsltTParamNamePart(
      std::shared_ptr<const DmngRsltTParamExpr> &&Param)
    : DmngRsltNamePart(DNPK_TemplateParam) {
    setTemplateParam(std::move(Param));
  }

  /// \brief Creates instance with new template parameter info (bind pointer).
  explicit DmngRsltTParamNamePart(const DmngRsltTParamExpr *Param)
    : DmngRsltNamePart(DNPK_TemplateParam) {
    setTemplateParam(Param);
  }


private:
  /// Template parameter expression.
  std::shared_ptr<const DmngRsltTParamExpr> TemplateParam;
};

/// \brief Name part that contains decltype() expression.
class DmngRsltDecltypeNamePart : public DmngRsltNamePart {
public:
  /// \brief Gets information about decltype() expression used as current
  ///        name part.
  std::shared_ptr<const DmngRsltDecltypeExpr> getDecltype() const {
    return Decltype;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltDecltypeNamePart>(*this);
  }

  // Manipulators.

  /// \brief Creates instance with decltype() expression info (by copy).
  explicit DmngRsltDecltypeNamePart(
      const std::shared_ptr<const DmngRsltDecltypeExpr> &Decltype)
    : DmngRsltNamePart(DNPK_Decltype), Decltype(Decltype) {
    assert(this->Decltype != nullptr && "Decltype expression must be defined.");
  }

  /// \brief Creates instance with decltype() expression info (by move).
  explicit DmngRsltDecltypeNamePart(
      std::shared_ptr<const DmngRsltDecltypeExpr> &&Decltype)
    : DmngRsltNamePart(DNPK_Decltype), Decltype(std::move(Decltype)) {
    assert(this->Decltype != nullptr && "Decltype expression must be defined.");
  }

  /// \brief Creates instance with decltype() expression info (bind pointer).
  explicit DmngRsltDecltypeNamePart(const DmngRsltDecltypeExpr *Decltype)
    : DmngRsltNamePart(DNPK_Decltype), Decltype(Decltype) {
    assert(this->Decltype != nullptr && "Decltype expression must be defined.");
  }


private:
  /// decltype() expression.
  std::shared_ptr<const DmngRsltDecltypeExpr> Decltype;
};


/// \brief Information about name parts (name prefix).
///
/// This collection is used only for substitution functionality.
/// It is not used in result.
class DmngRsltNameParts : public DmngRsltNode, public DmngRsltNamePartsBase {
public:
  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltNameParts>(*this);
  }

  /// \brief Creates copy of node with name parts (with optional
  ///        deep copy of parts).
  std::shared_ptr<DmngRsltNode> clone(bool DeepCopyNameParts) const {
    return std::make_shared<DmngRsltNameParts>(*this, DeepCopyNameParts);
  }


  /// \brief Creates instance with one name part (by copy).
  explicit DmngRsltNameParts(
      const std::shared_ptr<const DmngRsltNamePart>& Part)
    : DmngRsltNode(DNDK_NameParts), DmngRsltNamePartsBase(Part) {}

  /// \brief Creates instance with one name part (by move).
  explicit DmngRsltNameParts(std::shared_ptr<const DmngRsltNamePart>&& Part)
    : DmngRsltNode(DNDK_NameParts), DmngRsltNamePartsBase(Part) {}

  /// \brief Creates instance with one name part (bind pointer).
  explicit DmngRsltNameParts(const DmngRsltNamePart* Part)
    : DmngRsltNode(DNDK_NameParts), DmngRsltNamePartsBase(Part) {}

  /// \brief Creates instance with multiple parts (by copy).
  explicit DmngRsltNameParts(const DmngRsltNamePartsBase &NameParts)
    : DmngRsltNode(DNDK_NameParts), DmngRsltNamePartsBase(NameParts) {}

  /// \brief Creates instance with multiple parts (by move).
  explicit DmngRsltNameParts(DmngRsltNamePartsBase &&NameParts)
    : DmngRsltNode(DNDK_NameParts),
      DmngRsltNamePartsBase(std::move(NameParts)) {}

  /// \brief Creates instance that is a copy of other instance.
  ///
  /// \param Other    Other instance which will be copied.
  /// \param DeepCopy Indicates that name parts should be cloned as well.
  DmngRsltNameParts(const DmngRsltNameParts &Other, bool DeepCopy)
    : DmngRsltNode(Other), DmngRsltNamePartsBase(Other, DeepCopy) {}
};


/// \brief Demangled name information in result.
///
/// The class is intended for inheritance.
// Nodes: <encoding>
class DmngRsltName : public DmngRsltNode {
public:
  /// \brief Gets name kind.
  ///
  /// \return Enumeration which describes main kind of name.
  DmngNameKind getKind() const {
    return Kind;
  }

  /// \brief Gets name specific information.
  ///
  /// \tparam Kind Kind of name for which information will be retrieved.
  /// \return Specific name information, or empty shared pointer
  ///         if there is no information connected to selected Kind.
  template <DmngNameKind Kind>
  std::shared_ptr<const DmngRsltNameKSelT<Kind>> getAs() const {
    using DerivedT = DmngRsltNameKSelT<Kind>;

    if (Kind == this->Kind)
      return std::static_pointer_cast<const DerivedT>(getNode());
    return nullptr;
  }

  /// \brief Gets value indicating that name is ordinary function or data name.
  bool isOrdinary() const {
    return Kind == DNK_Ordinary;
  }

  /// \brief Gets value indicating that name is special.
  bool isSpecial() const {
    return Kind == DNK_Special;
  }

  /// \brief Gets value indicating that name refers to ordinary data.
  bool isOrdinaryData() const {
    return Kind == DNK_Ordinary && isData();
  }

  /// \brief Gets value indicating that name refers to special data
  ///        (virtual tables, type information, etc.).
  bool isSpecialData() const {
    return Kind == DNK_Special && isData();
  }

  /// \brief Gets value indicating that name refers to data.
  virtual bool isData() const = 0;

  /// \brief Gets value indicating that name refers to ordinary function.
  bool isOrdinaryFunction() const {
    return Kind == DNK_Ordinary && isFunction();
  }

  /// \brief Gets value indicating that name refers to special function
  ///        (virtual thunk, etc.).
  bool isSpecialFunction() const {
    return Kind == DNK_Special && isFunction();
  }

  /// \brief Gets value indicating that name refers to function.
  virtual bool isFunction() const = 0;

  // Manipulators.

  /// \brief Gets name specific information.
  ///
  /// \tparam Kind Kind of name for which information will be retrieved.
  /// \return Specific name information, or empty shared pointer
  ///         if there is no information connected to selected Kind.
  template <DmngNameKind Kind>
  std::shared_ptr<DmngRsltNameKSelT<Kind>> getAs() {
    using DerivedT = DmngRsltNameKSelT<Kind>;

    if (Kind == this->Kind)
      return std::static_pointer_cast<DerivedT>(getNode());
    return nullptr;
  }

protected:
  /// \brief Creates instance of base class with specific kind.
  explicit DmngRsltName(DmngNameKind Kind)
    : DmngRsltNode(DNDK_Name), Kind(Kind) {}


private:
  /// Main kind of name.
  DmngNameKind Kind;
};

/// \brief Information about ordinary (non-special) demangled name in result.
// Nodes: <name>, <unscoped-name>, <unscoped-template-name>, <nested-name>,
//        <local-name>
class DmngRsltOrdinaryName : public DmngRsltName,
                             public DmngRsltNamePartsBase,
                             public DmngRsltSignatureTypesBase,
                             public DmngRsltVendorQualsBase {
public:
  /// \brief Gets value indicating that name refers to (ordinary) data.
  bool isData() const override {
    return getSignatureTypes().empty();
  }

  /// \brief Gets value indicating that name refers to (ordinary) function.
  bool isFunction() const override {
    return !getSignatureTypes().empty();
  }


  /// \brief Indicates that name is local.
  bool isLocal() const {
    return LocalScope != nullptr;
  }

  /// \brief Gets local scope for local name.
  ///
  /// \return Name of local scope, or empty shared pointer if name is not local.
  std::shared_ptr<const DmngRsltName> getLocalScope() const {
    return LocalScope;
  }

  /// \brief Gets discriminator index in local scope of entity.
  ///
  /// Gets 0-based index in local scope for entities with the same name located
  /// in unnamed subscopes of the same local scope (in lexical order).
  /// If entity is string literal the index discrimanates literals that have
  /// different values.
  ///
  /// \return 0-based index, or -1 if current name is not local name.
  int getInLocalScopeIdx() const {
    return isLocal() ? InLocalScopeIdx : -1;
  }

  /// \brief Gets index of parameter (in reversed order) which default value
  ///        contains entity described by current name (usually closure type).
  ///
  /// Gets 0-based index of parameter in reverse order (0 refers to last
  /// parameter, 1 - to second to last, etc.). If current name does not refer
  /// to default value, the -1 is returned instead.
  ///
  /// If index is valid (non-negative), getInLocalScopeIdx() treates it as
  /// another qualification of local scope (so the returned in-scope index is
  /// local to function parameter).
  ///
  /// \return 0-based index (reverse order), or -1 if name does not refer
  ///         to default value.
  int getDefaultValueParamRIdx() const {
    return isLocal() ? DefaultValueParamRIdx : -1;
  }

  /// \brief Indicates that current name refers to string literal.
  bool isStringLiteral() const {
    return IsStringLiteral;
  }


  /// \brief Gets cvr-qualifiers for current name (usually used with member
  ///        functions).
  DmngCvrQuals getCvrQuals() const {
    return CvrQuals;
  }

  /// \brief Gets reference qualifier for current name (usually used with member
  ///        functions).
  DmngRefQuals getRefQuals() const {
    return RefQuals;
  }


  /// \brief Gets value indicating that current (function) name has return
  ///        type encoded in signature.
  bool hasReturnType() const override {
    if (getSignatureTypes().empty() || getParts().empty())
      return false;

    const auto& LPart = getParts().back();
    if(LPart == nullptr || !LPart->isTemplate() ||
       LPart->getPartKind() == DNPK_Constructor ||
       LPart->getPartKind() == DNPK_Destructor)
      return false;

    const auto& LPartAsOp = LPart->getAs<DNPK_Operator>();
    return LPartAsOp == nullptr || !LPartAsOp->isConversionOperator();
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltOrdinaryName>(*this);
  }

  /// \brief Creates copy of node with name parts (with optional
  ///        deep copy of parts).
  std::shared_ptr<DmngRsltOrdinaryName> clone(bool DeepCopyNameParts) const {
    return std::make_shared<DmngRsltOrdinaryName>(*this, DeepCopyNameParts);
  }

  // Manipulators.

  /// \brief Sets name that identifies local scope (by copy).
  void setLocalScope(const std::shared_ptr<const DmngRsltName> &ScopeName) {
    LocalScope = ScopeName;
  }

  /// \brief Sets name that identifies local scope (by move).
  void setLocalScope(std::shared_ptr<const DmngRsltName> &&ScopeName) {
    LocalScope = std::move(ScopeName);
  }

  /// \brief Sets name that identifies local scope (bind pointer).
  void setLocalScope(const DmngRsltName *ScopeName) {
    LocalScope = std::shared_ptr<const DmngRsltName>(ScopeName);
  }

  /// \brief Sets discriminator index in local scope.
  ///
  /// If Idx is negative, the function asserts and sets it to 0.
  void setInLocalScopeIdx(int Idx) {
    assert(InLocalScopeIdx >= 0 &&
           "Local index must be non-negative for valid local name.");
    InLocalScopeIdx = Idx >= 0 ? Idx : 0;
  }

  /// \brief Sets index (in reversed order) of parameter which default value
  ///        forms the scope for current name.
  ///
  /// \param ReversedIdx Index (in reversed order) of parameter. If index
  ///        is negative it is set -1 (not in parameters).
  void setDefaultValueParamRIdx(int ReversedIdx) {
    DefaultValueParamRIdx = ReversedIdx >= 0 ? ReversedIdx : -1;
  }

  /// \brief Sets value indicating that current name refers to string literal.
  void setIsStringLiteral(bool IsStringLiteral) {
    this->IsStringLiteral = IsStringLiteral;
  }

  /// \brief Sets cvr-qualifiers (const, volatile, restrict, etc.).
  void setCvrQuals(DmngCvrQuals Quals) {
    CvrQuals = Quals;
  }

  /// \brief Sets reference qualifiers (&, &&, etc.).
  void setRefQuals(DmngRefQuals Quals) {
    RefQuals = Quals;
  }

  /// \brief Resets all qualifiers.
  void resetQuals() {
    resetVendorQuals();
    setCvrQuals(DCVQ_None);
    setRefQuals(DRQ_None);
  }

private:
  /// \brief Creates instance of ordinary name (for function or data).
  explicit DmngRsltOrdinaryName(DmngRsltNamePartsBase &&NameParts)
    : DmngRsltName(DNK_Ordinary), DmngRsltNamePartsBase(std::move(NameParts)),
    InLocalScopeIdx(0), DefaultValueParamRIdx(-1),
    IsStringLiteral(false), CvrQuals(DCVQ_None), RefQuals(DRQ_None) {}
public:
  /// \brief Creates instance of ordinary name (for function or data) with one
  ///        part (by copy).
  explicit DmngRsltOrdinaryName(
      const std::shared_ptr<const DmngRsltNamePart> &Part)
    : DmngRsltOrdinaryName(DmngRsltNameParts(Part)) {}

  /// \brief Creates instance of ordinary name (for function or data) with one
  ///        part (by move).
  explicit DmngRsltOrdinaryName(std::shared_ptr<const DmngRsltNamePart> &&Part)
    : DmngRsltOrdinaryName(DmngRsltNameParts(std::move(Part))) {}

  /// \brief Creates instance of ordinary name (for function or data) with one
  ///        part (bind pointer).
  explicit DmngRsltOrdinaryName(const DmngRsltNamePart *Part)
    : DmngRsltOrdinaryName(DmngRsltNameParts(Part)) {}

  /// \brief Creates instance of ordinary name (for function or data) with
  ///        multiple parts (by copy).
  explicit DmngRsltOrdinaryName(
      const std::shared_ptr<const DmngRsltNameParts> &Parts)
    : DmngRsltOrdinaryName(Parts != nullptr ? DmngRsltNameParts(*Parts) :
                                              DmngRsltNameParts(nullptr)) {}

  /// \brief Creates instance of ordinary name for string literal (no parts).
  DmngRsltOrdinaryName()
    : DmngRsltName(DNK_Ordinary), InLocalScopeIdx(0), DefaultValueParamRIdx(-1),
    IsStringLiteral(true), CvrQuals(DCVQ_None), RefQuals(DRQ_None) {}

  /// \brief Creates instance that is a copy of other instance.
  ///
  /// \param Other             Other instance which will be copied.
  /// \param DeepCopyNameParts Indicates that name parts should be cloned as
  ///                          well.
  DmngRsltOrdinaryName(const DmngRsltOrdinaryName &Other,
                       bool DeepCopyNameParts)
    : DmngRsltName(Other), DmngRsltNamePartsBase(Other, DeepCopyNameParts),
      DmngRsltSignatureTypesBase(Other), DmngRsltVendorQualsBase(Other),
      LocalScope(Other.LocalScope), InLocalScopeIdx(Other.InLocalScopeIdx),
      DefaultValueParamRIdx(Other.DefaultValueParamRIdx),
      IsStringLiteral(Other.IsStringLiteral), CvrQuals(Other.CvrQuals),
      RefQuals(Other.RefQuals) {}


private:
  /// Name of local scope in which current name is located (usually function
  /// name.
  std::shared_ptr<const DmngRsltName> LocalScope;
  /// 0-based index in local scope (to discriminate similarly named declarations
  /// from unnamed sub-scopes). Index is applied in lexical order.
  int InLocalScopeIdx;
  /// 0-based index of parameter (in reversed order) which default value
  /// contains entity referred by current name (usually closure type).
  int DefaultValueParamRIdx;
  /// Indicates that current name relates to string literal.
  bool IsStringLiteral;

  /// cvr-qualifiers.
  DmngCvrQuals CvrQuals;
  /// Reference qualifiers.
  DmngRefQuals RefQuals;
};

/// \brief Information about special demangled name in result.
// Nodes: <special-name>
class DmngRsltSpecialName : public DmngRsltName {
public:
  /// Type of information used to describe related objects (sub-type
  /// of related names).
  using RelatedObjT = DmngRsltOrdinaryName;


  /// \brief Gets sub-kind of special name.
  DmngSpecialNameKind getSpecialKind() const {
    return SpecialKind;
  }

  /// \brief Gets value indicating that name refers to (special) data
  ///        (virtual tables, type information, etc.).
  bool isData() const override {
    switch (SpecialKind) {
    case DSNK_VirtualTable:
    case DSNK_VirtualTableTable:
    case DSNK_TypeInfoStruct:
    case DSNK_TypeInfoNameString:
    case DSNK_GuardVariable:
    case DSNK_LifeExtTemporary:
      return true;
    default:
      return false;
    }
  }

  /// \brief Gets value indicating that name refers to special function
  ///        (virtual thunk, etc.).
  bool isFunction() const override {
    switch (SpecialKind) {
    case DSNK_VirtualThunk:
      return true;
    default:
      return false;
    }
  }

  /// \brief Indicates that name is connected to virtual functions,
  ///        virtual inheritance and/or polymorphism.
  bool isVirtual() const {
    switch (SpecialKind) {
    case DSNK_VirtualTable:
    case DSNK_VirtualTableTable:
    case DSNK_VirtualThunk:
      return true;
    default:
      return false;
    }
  }

  /// \brief Indicates that name is connected to runtime type information,
  ///        type_info structures, etc.
  bool isRtti() const {
    switch (SpecialKind) {
    case DSNK_TypeInfoStruct:
    case DSNK_TypeInfoNameString:
      return true;
    default:
      return false;
    }
  }


  /// \brief Gets related type connected to special data (virtual table, etc.).
  ///
  /// Gets name of the type for which current special entity was created.
  ///
  /// \brief Name of the related type, or nullptr if special entity does not
  ///        relate to any type.
  std::shared_ptr<const DmngRsltType> getRelatedType() const {
    switch (SpecialKind) {
    case DSNK_VirtualTable:
    case DSNK_VirtualTableTable:
    case DSNK_TypeInfoStruct:
    case DSNK_TypeInfoNameString:
      return RelatedType;
    default:
      return nullptr;
    }
  }


  /// \brief Gets origin function or data delegated/accessed via virtual thunk.
  ///
  /// \return Name of origin object, or nullptr if current special entity
  ///         is not thunk.
  std::shared_ptr<const DmngRsltName> getOrigin() const {
    switch (SpecialKind) {
    case DSNK_VirtualThunk:
      return RelatedName;
    default:
      return nullptr;
    }
  }

  /// \brief Gets adjustment information for this pointer (for thunk).
  const DmngRsltAdjustOffset &getThisAdjustment() const {
    return ThisAdjustment;
  }

  /// \brief Gets adjustment information for return type (for thunk when
  ///        covariant return is used).
  const DmngRsltAdjustOffset &getReturnAdjustment() const {
    return ReturnAdjustment;
  }


  /// \brief Gets object related to life-extended temporary or guard variable.
  ///
  /// \return Ordinary name of object related to current special entity, or
  ///         nullptr if special entity is not temporary nor guard variable.
  std::shared_ptr<const RelatedObjT>
  getRelatedObject() const {
    switch (SpecialKind) {
    case DSNK_GuardVariable:
    case DSNK_LifeExtTemporary:
      if (RelatedName != nullptr)
        return RelatedName->getAs<DNK_Ordinary>();
    default:
      return nullptr;
    }
  }

  /// \brief Gets identifier of life-extended temporary (to discriminate beteen
  ///        them when they all relate to single object).
  ///
  /// \return Discriminator number for temporaries.
  unsigned long long getId() const {
    return Id;
  }

  /// \brief Creates shallow copy of node (or any derived type of node).
  std::shared_ptr<DmngRsltNode> clone() const override {
    return std::make_shared<DmngRsltSpecialName>(*this);
  }

  // Manipulators.

private:
  /// \brief Creates new instance for special name information (unified ctor).
  DmngRsltSpecialName(DmngSpecialNameKind SpecialKind,
                      const std::shared_ptr<const DmngRsltType> &RelType,
                      const std::shared_ptr<const DmngRsltName> &RelName,
                      const DmngRsltAdjustOffset &ThisAdjustment,
                      const DmngRsltAdjustOffset &ReturnAdjustment,
                      unsigned long long Id)
    : DmngRsltName(DNK_Special), SpecialKind(SpecialKind),
      RelatedType(RelType), RelatedName(RelName),
      ThisAdjustment(ThisAdjustment), ReturnAdjustment(ReturnAdjustment),
      Id(Id) {}

public:
  /// \brief Creates special name as virtual table for specific type.
  static DmngRsltSpecialName createVirtualTable(
      const std::shared_ptr<const DmngRsltType> &VTableType) {
    assert(VTableType != nullptr && "Related type must exist.");
    return DmngRsltSpecialName(DSNK_VirtualTable, VTableType, nullptr,
                               DmngRsltAdjustOffset(), DmngRsltAdjustOffset(),
                               0);
  }

  /// \brief Creates special name as VTT for specific type.
  static DmngRsltSpecialName createVirtualTableTable(
      const std::shared_ptr<const DmngRsltType> &VTTType) {
    assert(VTTType != nullptr && "Related type must exist.");
    return DmngRsltSpecialName(DSNK_VirtualTableTable, VTTType, nullptr,
                               DmngRsltAdjustOffset(), DmngRsltAdjustOffset(),
                               0);
  }

  /// \brief Creates special name as type_info structure for specific type.
  static DmngRsltSpecialName createTypeInfoStruct(
      const std::shared_ptr<const DmngRsltType> &DescribedType) {
    assert(DescribedType != nullptr && "Related type must exist.");
    return DmngRsltSpecialName(DSNK_TypeInfoStruct, DescribedType, nullptr,
                               DmngRsltAdjustOffset(), DmngRsltAdjustOffset(),
                               0);
  }

  /// \brief Creates special name as type_info name string for specific type.
  static DmngRsltSpecialName createTypeInfoNameString(
      const std::shared_ptr<const DmngRsltType> &DescribedType) {
    assert(DescribedType != nullptr && "Related type must exist.");
    return DmngRsltSpecialName(DSNK_TypeInfoNameString, DescribedType, nullptr,
                               DmngRsltAdjustOffset(), DmngRsltAdjustOffset(),
                               0);
  }

  /// \brief Creates special name as virtual thunk delegating to origin data
  ///        or function and adjusting "this" pointer to proper base.
  static DmngRsltSpecialName createVirtualThunk(
      const std::shared_ptr<const DmngRsltName> &Origin,
      const DmngRsltAdjustOffset &ThisAdjustment) {
    assert(Origin != nullptr && "Origin of thunk must exist.");
    return DmngRsltSpecialName(DSNK_VirtualThunk, nullptr, Origin,
                               ThisAdjustment, DmngRsltAdjustOffset(), 0);
  }

  /// \brief Creates special name as virtual thunk delegating to origin data
  ///        or function, adjusting "this" pointer to proper base and adjusting
  ///        return object as well (return value covariance).
  static DmngRsltSpecialName createVirtualThunk(
      const std::shared_ptr<const DmngRsltName> &Origin,
      const DmngRsltAdjustOffset &ThisAdjustment,
      const DmngRsltAdjustOffset &ReturnAdjustment) {
    return DmngRsltSpecialName(DSNK_VirtualThunk, nullptr, Origin,
                               ThisAdjustment, ReturnAdjustment, 0);
  }

  /// \brief Creates special name as guard variable for one-time initialization
  ///        of releated object (inline static initialization).
  static DmngRsltSpecialName createGuardVariable(
      const std::shared_ptr<const RelatedObjT> &GuardedObj) {
    assert(GuardedObj != nullptr && "Related object must exist.");
    return DmngRsltSpecialName(DSNK_GuardVariable, nullptr, GuardedObj,
                               DmngRsltAdjustOffset(), DmngRsltAdjustOffset(),
                               0);
  }

  /// \brief Formats special name as life-extended temporary binded directly
  ///        or indirectly to related static object (its reference).
  ///
  /// \param RelatedObj Related object which extends lifetime of temporary.
  /// \param Id         Discriminator value (allows to differentiate two
  ///                   temporaries binded to the same object).
  static DmngRsltSpecialName createLifeExtTemporary(
      const std::shared_ptr<const RelatedObjT> &RelatedObj,
      unsigned long long Id = 0) {
    assert(RelatedObj != nullptr && "Related object must exist.");
    return DmngRsltSpecialName(DSNK_LifeExtTemporary, nullptr, RelatedObj,
                               DmngRsltAdjustOffset(), DmngRsltAdjustOffset(),
                               Id);
  }


private:
  /// Sub-kind of special name.
  DmngSpecialNameKind SpecialKind;

  /// Type described by special data name.
  std::shared_ptr<const DmngRsltType> RelatedType;
  /// Name of related or origin entity (data or function).
  ///
  /// The name can identify function which is delegated via virtual thunk,
  /// object guarded by guard variable or object which temporary is bound to.
  std::shared_ptr<const DmngRsltName> RelatedName;

  /// Adjustment for this pointer in virtual thunk.
  DmngRsltAdjustOffset ThisAdjustment;
  /// Adjustment for return type in virtual thunk (return type covariance).
  DmngRsltAdjustOffset ReturnAdjustment;

  /// Identifier that allows to differentiate between life-extended temporaries
  /// connected to the same object (in case of partial/multi-level
  /// initialization).
  unsigned long long Id;
};


/// \brief Result of name parsing from C++ demangler (clang OCL C++ flavor).
///
/// Represents result returned from demangling.
class DmngRslt {
public:
  /// \brief Gets "mangled" name.
  ///
  /// \return Original name before mangling.
  const std::string& getMangledName() const {
    return MangledName;
  }

  /// \brief Gets demangled name information.
  ///
  /// \return Structure that describes demangled name with all additional
  ///         information.
  std::shared_ptr<const DmngRsltName> getName() const {
    return DemangledName;
  }

  /// \brief Gets value indicating that demangling was successful.
  bool isSuccessful() const {
    return IsSuccessful;
  }

  /// \brief Gets value indicating that demangling has failed.
  bool isFailed() const {
    return !IsSuccessful;
  }

  /// \brief Returns the same as isSuccessful().
  explicit operator bool() const
  {
    return IsSuccessful;
  }

  /// \brief Returns the same as isFailed().
  bool operator !() const
  {
    return !IsSuccessful;
  }

  // Manipulators.

  /// \brief Marks result as failed.
  ///
  /// Result marked as failed indicates failure of entire parsing. After
  /// marking collection as failed, it success state cannot be changed.
  DmngRslt &setFailed() {
    IsSuccessful = false;
    return *this;
  }

  /// \brief Sets demangled name (by copy).
  void setName(const std::shared_ptr<const DmngRsltName> &Name) {
    assert(Name != nullptr && "Name must exist (in order to set).");
    DemangledName = Name;
  }

  /// \brief Sets demangled name (by move).
  void setName(std::shared_ptr<const DmngRsltName> &&Name) {
    assert(Name != nullptr && "Name must exist (in order to set).");
    DemangledName = std::move(Name);
  }

  /// \brief Sets demangled name (bind pointer).
  void setName(const DmngRsltName *Name) {
    assert(Name != nullptr && "Name must exist (in order to set).");
    DemangledName = std::shared_ptr<const DmngRsltName>(Name);
  }

  /// \brief Constructs result of parsing done by demangler.
  ///
  /// \param MangledName Name that will be demangled (via copy).
  explicit DmngRslt(const std::string &MangledName)
    : MangledName(MangledName), IsSuccessful(true) {}
  /// \brief Constructs result of parsing done by demangler.
  ///
  /// \param MangledName Name that will be demangled (via move).
  explicit DmngRslt(std::string &&MangledName)
    : MangledName(std::move(MangledName)), IsSuccessful(true) {}


private:
  /// Original name (name to parse).
  std::string MangledName;
  /// Information about demangled name.
  std::shared_ptr<const DmngRsltName> DemangledName;
  bool IsSuccessful;
};


// -----------------------------------------------------------------------------
// IMPLEMENTATION (WHEN ALL RESULT OBJECTS ARE COMPLETE)
// -----------------------------------------------------------------------------

inline DmngRsltNamePartsBase::DmngRsltNamePartsBase(
    const DmngRsltNamePartsBase& Other, bool DeepCopy) {
  if (!DeepCopy) {
    Parts = Other.Parts;
    return;
  }

  for (const auto &Part : Other.Parts) {
    Parts.push_back(Part->clone()->getAs<DNDK_NamePart>());
  }
}

inline bool DmngRsltTArg::isExpression() const {
  return Value != nullptr && Value->getNodeKind() == DNDK_Expr;
}

inline bool DmngRsltTArg::isType() const {
  return Value != nullptr && Value->getNodeKind() == DNDK_Type;
}

inline bool DmngRsltTArg::isPack() const {
  return Value == nullptr;
}

inline std::shared_ptr<const DmngRsltExpr> DmngRsltTArg::getExpression() const {
  return Value == nullptr ? nullptr : Value->getAs<DNDK_Expr>();
}

inline std::shared_ptr<const DmngRsltType> DmngRsltTArg::getType() const {
  return Value == nullptr ? nullptr : Value->getAs<DNDK_Type>();
}

inline std::shared_ptr<const DmngRsltNode> DmngRsltTArg::getValue() const {
  return Value;
}

inline const DmngRsltTArg::TArgsPackColT &DmngRsltTArg::getPack() const {
  return Value == nullptr ? Pack : EmpyPack;
}

inline void DmngRsltTArg::addPackArg(const DmngRsltTArg &Arg) {
  assert(Value == nullptr &&
         "Adding pack element to non-pack template argument.");
  Pack.push_back(Arg);
}

inline void DmngRsltTArg::addPackArg(DmngRsltTArg &&Arg) {
  assert(Value == nullptr &&
         "Adding pack element to non-pack template argument.");
  Pack.push_back(std::move(Arg));
}

inline DmngRsltTArg::DmngRsltTArg(
    const std::shared_ptr<const DmngRsltExpr> &Value)
  : Value(Value) {
  assert(this->Value != nullptr && "Template arg expression must exist.");
}

inline DmngRsltTArg::DmngRsltTArg(std::shared_ptr<const DmngRsltExpr> &&Value)
  : Value(std::move(Value)) {
  assert(this->Value != nullptr && "Template arg expression must exist.");
}

inline DmngRsltTArg::DmngRsltTArg(
    const std::shared_ptr<const DmngRsltType> &Value)
  : Value(Value) {
  assert(this->Value != nullptr && "Template arg type must exist.");
}

inline DmngRsltTArg::DmngRsltTArg(std::shared_ptr<const DmngRsltType> &&Value)
  : Value(std::move(Value)) {
  assert(this->Value != nullptr && "Template arg type must exist.");
}

} // oclcxx_adaptation
} // spirv

#endif // CLANG_LIB_CODEGEN_OCLCXXREWRITE_OCLCXXDEMANGLERRESULT_H
