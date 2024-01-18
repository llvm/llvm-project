//===-- include/flang/Semantics/type.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_TYPE_H_
#define FORTRAN_SEMANTICS_TYPE_H_

#include "flang/Common/Fortran.h"
#include "flang/Common/idioms.h"
#include "flang/Evaluate/expression.h"
#include "flang/Parser/char-block.h"
#include <algorithm>
#include <iosfwd>
#include <map>
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace Fortran::parser {
struct Keyword;
}

namespace Fortran::semantics {

class Scope;
class SemanticsContext;
class Symbol;

/// A SourceName is a name in the cooked character stream,
/// i.e. a range of lower-case characters with provenance.
using SourceName = parser::CharBlock;
using TypeCategory = common::TypeCategory;
using SomeExpr = evaluate::Expr<evaluate::SomeType>;
using MaybeExpr = std::optional<SomeExpr>;
using SomeIntExpr = evaluate::Expr<evaluate::SomeInteger>;
using MaybeIntExpr = std::optional<SomeIntExpr>;
using SubscriptIntExpr = evaluate::Expr<evaluate::SubscriptInteger>;
using MaybeSubscriptIntExpr = std::optional<SubscriptIntExpr>;
using KindExpr = SubscriptIntExpr;

// An array spec bound: an explicit integer expression, assumed size
// or implied shape(*), or assumed or deferred shape(:).  In the absence
// of explicit lower bounds it is not possible to distinguish assumed
// shape bounds from deferred shape bounds without knowing whether the
// particular symbol is an allocatable/pointer or a non-allocatable
// non-pointer dummy; use the symbol-based predicates for those
// determinations.
class Bound {
public:
  static Bound Star() { return Bound(Category::Star); }
  static Bound Colon() { return Bound(Category::Colon); }
  explicit Bound(MaybeSubscriptIntExpr &&expr) : expr_{std::move(expr)} {}
  explicit Bound(common::ConstantSubscript bound);
  Bound(const Bound &) = default;
  Bound(Bound &&) = default;
  Bound &operator=(const Bound &) = default;
  Bound &operator=(Bound &&) = default;
  bool isExplicit() const { return category_ == Category::Explicit; }
  bool isStar() const { return category_ == Category::Star; }
  bool isColon() const { return category_ == Category::Colon; }
  MaybeSubscriptIntExpr &GetExplicit() { return expr_; }
  const MaybeSubscriptIntExpr &GetExplicit() const { return expr_; }
  void SetExplicit(MaybeSubscriptIntExpr &&expr) {
    CHECK(isExplicit());
    expr_ = std::move(expr);
  }

private:
  enum class Category { Explicit, Star, Colon };
  Bound(Category category) : category_{category} {}
  Bound(Category category, MaybeSubscriptIntExpr &&expr)
      : category_{category}, expr_{std::move(expr)} {}
  Category category_{Category::Explicit};
  MaybeSubscriptIntExpr expr_;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const Bound &);
};

// A type parameter value: integer expression, assumed/implied(*),
// or deferred(:).
class ParamValue {
public:
  static ParamValue Assumed(common::TypeParamAttr attr) {
    return ParamValue{Category::Assumed, attr};
  }
  static ParamValue Deferred(common::TypeParamAttr attr) {
    return ParamValue{Category::Deferred, attr};
  }
  ParamValue(const ParamValue &) = default;
  explicit ParamValue(MaybeIntExpr &&, common::TypeParamAttr);
  explicit ParamValue(SomeIntExpr &&, common::TypeParamAttr attr);
  explicit ParamValue(common::ConstantSubscript, common::TypeParamAttr attr);
  bool isExplicit() const { return category_ == Category::Explicit; }
  bool isAssumed() const { return category_ == Category::Assumed; }
  bool isDeferred() const { return category_ == Category::Deferred; }
  const MaybeIntExpr &GetExplicit() const { return expr_; }
  void SetExplicit(SomeIntExpr &&);
  bool isKind() const { return attr_ == common::TypeParamAttr::Kind; }
  bool isLen() const { return attr_ == common::TypeParamAttr::Len; }
  void set_attr(common::TypeParamAttr attr) { attr_ = attr; }
  bool operator==(const ParamValue &that) const {
    return category_ == that.category_ && expr_ == that.expr_;
  }
  bool operator!=(const ParamValue &that) const { return !(*this == that); }
  std::string AsFortran() const;

private:
  enum class Category { Explicit, Deferred, Assumed };
  ParamValue(Category category, common::TypeParamAttr attr)
      : category_{category}, attr_{attr} {}
  Category category_{Category::Explicit};
  common::TypeParamAttr attr_{common::TypeParamAttr::Kind};
  MaybeIntExpr expr_;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ParamValue &);
};

class IntrinsicTypeSpec {
public:
  TypeCategory category() const { return category_; }
  const KindExpr &kind() const { return kind_; }
  bool operator==(const IntrinsicTypeSpec &x) const {
    return category_ == x.category_ && kind_ == x.kind_;
  }
  bool operator!=(const IntrinsicTypeSpec &x) const { return !operator==(x); }
  std::string AsFortran() const;

protected:
  IntrinsicTypeSpec(TypeCategory, KindExpr &&);

private:
  TypeCategory category_;
  KindExpr kind_;
  friend llvm::raw_ostream &operator<<(
      llvm::raw_ostream &os, const IntrinsicTypeSpec &x);
};

class NumericTypeSpec : public IntrinsicTypeSpec {
public:
  NumericTypeSpec(TypeCategory category, KindExpr &&kind)
      : IntrinsicTypeSpec(category, std::move(kind)) {
    CHECK(common::IsNumericTypeCategory(category));
  }
};

class LogicalTypeSpec : public IntrinsicTypeSpec {
public:
  explicit LogicalTypeSpec(KindExpr &&kind)
      : IntrinsicTypeSpec(TypeCategory::Logical, std::move(kind)) {}
};

class CharacterTypeSpec : public IntrinsicTypeSpec {
public:
  CharacterTypeSpec(ParamValue &&length, KindExpr &&kind)
      : IntrinsicTypeSpec(TypeCategory::Character, std::move(kind)),
        length_{std::move(length)} {}
  const ParamValue &length() const { return length_; }
  bool operator==(const CharacterTypeSpec &that) const {
    return kind() == that.kind() && length_ == that.length_;
  }
  std::string AsFortran() const;

private:
  ParamValue length_;
  friend llvm::raw_ostream &operator<<(
      llvm::raw_ostream &os, const CharacterTypeSpec &x);
};

class ShapeSpec {
public:
  // lb:ub
  static ShapeSpec MakeExplicit(Bound &&lb, Bound &&ub) {
    return ShapeSpec(std::move(lb), std::move(ub));
  }
  // 1:ub
  static const ShapeSpec MakeExplicit(Bound &&ub) {
    return MakeExplicit(Bound{1}, std::move(ub));
  }
  // 1:
  static ShapeSpec MakeAssumedShape() {
    return ShapeSpec(Bound{1}, Bound::Colon());
  }
  // lb:
  static ShapeSpec MakeAssumedShape(Bound &&lb) {
    return ShapeSpec(std::move(lb), Bound::Colon());
  }
  // :
  static ShapeSpec MakeDeferred() {
    return ShapeSpec(Bound::Colon(), Bound::Colon());
  }
  // 1:*
  static ShapeSpec MakeImplied() { return ShapeSpec(Bound{1}, Bound::Star()); }
  // lb:*
  static ShapeSpec MakeImplied(Bound &&lb) {
    return ShapeSpec(std::move(lb), Bound::Star());
  }
  // ..
  static ShapeSpec MakeAssumedRank() {
    return ShapeSpec(Bound::Star(), Bound::Star());
  }

  ShapeSpec(const ShapeSpec &) = default;
  ShapeSpec(ShapeSpec &&) = default;
  ShapeSpec &operator=(const ShapeSpec &) = default;
  ShapeSpec &operator=(ShapeSpec &&) = default;

  Bound &lbound() { return lb_; }
  const Bound &lbound() const { return lb_; }
  Bound &ubound() { return ub_; }
  const Bound &ubound() const { return ub_; }

private:
  ShapeSpec(Bound &&lb, Bound &&ub) : lb_{std::move(lb)}, ub_{std::move(ub)} {}
  Bound lb_;
  Bound ub_;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ShapeSpec &);
};

struct ArraySpec : public std::vector<ShapeSpec> {
  ArraySpec() {}
  int Rank() const { return size(); }
  // These names are not exclusive, as some categories cannot be
  // distinguished without knowing whether the particular symbol
  // is allocatable, pointer, or a non-allocatable non-pointer dummy.
  // Use the symbol-based predicates for exact results.
  inline bool IsExplicitShape() const;
  inline bool CanBeAssumedShape() const;
  inline bool CanBeDeferredShape() const;
  inline bool CanBeImpliedShape() const;
  inline bool CanBeAssumedSize() const;
  inline bool IsAssumedRank() const;

private:
  // Check non-empty and predicate is true for each element.
  template <typename P> bool CheckAll(P predicate) const {
    return !empty() && std::all_of(begin(), end(), predicate);
  }
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const ArraySpec &);

// Each DerivedTypeSpec has a typeSymbol that has DerivedTypeDetails.
// The name may not match the symbol's name in case of a USE rename.
class DerivedTypeSpec {
public:
  enum class Category { DerivedType, IntrinsicVector, PairVector, QuadVector };

  using RawParameter = std::pair<const parser::Keyword *, ParamValue>;
  using RawParameters = std::vector<RawParameter>;
  using ParameterMapType = std::map<SourceName, ParamValue>;
  DerivedTypeSpec(SourceName, const Symbol &);
  DerivedTypeSpec(const DerivedTypeSpec &);
  DerivedTypeSpec(DerivedTypeSpec &&);

  const SourceName &name() const { return name_; }
  const Symbol &typeSymbol() const { return typeSymbol_; }
  const Scope *scope() const { return scope_; }
  // Return scope_ if it is set, or the typeSymbol_ scope otherwise.
  const Scope *GetScope() const;
  void set_scope(const Scope &);
  void ReplaceScope(const Scope &);
  const RawParameters &rawParameters() const { return rawParameters_; }
  const ParameterMapType &parameters() const { return parameters_; }

  bool MightBeParameterized() const;
  bool IsForwardReferenced() const;
  bool HasDefaultInitialization(
      bool ignoreAllocatable = false, bool ignorePointer = true) const;
  bool HasDestruction() const;

  // The "raw" type parameter list is a simple transcription from the
  // parameter list in the parse tree, built by calling AddRawParamValue().
  // It can be used with forward-referenced derived types.
  void AddRawParamValue(const parser::Keyword *, ParamValue &&);
  // Checks the raw parameter list against the definition of a derived type.
  // Converts the raw parameter list to a map, naming each actual parameter.
  void CookParameters(evaluate::FoldingContext &);
  // Evaluates type parameter expressions.
  void EvaluateParameters(SemanticsContext &);
  void AddParamValue(SourceName, ParamValue &&);
  // Creates a Scope for the type and populates it with component
  // instantiations that have been specialized with actual type parameter
  // values, which are cooked &/or evaluated if necessary.
  void Instantiate(Scope &containingScope);

  ParamValue *FindParameter(SourceName);
  const ParamValue *FindParameter(SourceName target) const {
    auto iter{parameters_.find(target)};
    if (iter != parameters_.end()) {
      return &iter->second;
    } else {
      return nullptr;
    }
  }
  bool operator==(const DerivedTypeSpec &that) const {
    return RawEquals(that) && parameters_ == that.parameters_;
  }
  bool operator!=(const DerivedTypeSpec &that) const {
    return !(*this == that);
  }
  // For TYPE IS & CLASS IS: kind type parameters must be
  // explicit and equal, len type parameters are ignored.
  bool Match(const DerivedTypeSpec &) const;
  std::string AsFortran() const;
  std::string VectorTypeAsFortran() const;

  Category category() const { return category_; }
  void set_category(Category category) { category_ = category; }
  bool IsVectorType() const {
    return category_ == Category::IntrinsicVector ||
        category_ == Category::PairVector || category_ == Category::QuadVector;
  }

private:
  SourceName name_;
  const Symbol &typeSymbol_;
  const Scope *scope_{nullptr}; // same as typeSymbol_.scope() unless PDT
  bool cooked_{false};
  bool evaluated_{false};
  bool instantiated_{false};
  RawParameters rawParameters_;
  ParameterMapType parameters_;
  Category category_{Category::DerivedType};
  bool RawEquals(const DerivedTypeSpec &that) const {
    return &typeSymbol_ == &that.typeSymbol_ && cooked_ == that.cooked_ &&
        rawParameters_ == that.rawParameters_;
  }
  friend llvm::raw_ostream &operator<<(
      llvm::raw_ostream &, const DerivedTypeSpec &);
};

class DeclTypeSpec {
public:
  enum Category {
    Numeric,
    Logical,
    Character,
    TypeDerived,
    ClassDerived,
    TypeStar,
    ClassStar
  };

  // intrinsic-type-spec or TYPE(intrinsic-type-spec), not character
  DeclTypeSpec(NumericTypeSpec &&);
  DeclTypeSpec(LogicalTypeSpec &&);
  // character
  DeclTypeSpec(const CharacterTypeSpec &);
  DeclTypeSpec(CharacterTypeSpec &&);
  // TYPE(derived-type-spec) or CLASS(derived-type-spec)
  DeclTypeSpec(Category, const DerivedTypeSpec &);
  DeclTypeSpec(Category, DerivedTypeSpec &&);
  // TYPE(*) or CLASS(*)
  DeclTypeSpec(Category);

  bool operator==(const DeclTypeSpec &) const;
  bool operator!=(const DeclTypeSpec &that) const { return !operator==(that); }

  Category category() const { return category_; }
  void set_category(Category category) { category_ = category; }
  bool IsPolymorphic() const {
    return category_ == ClassDerived || IsUnlimitedPolymorphic();
  }
  bool IsUnlimitedPolymorphic() const {
    return category_ == TypeStar || category_ == ClassStar;
  }
  bool IsAssumedType() const { return category_ == TypeStar; }
  bool IsNumeric(TypeCategory) const;
  bool IsSequenceType() const;
  const NumericTypeSpec &numericTypeSpec() const;
  const LogicalTypeSpec &logicalTypeSpec() const;
  const CharacterTypeSpec &characterTypeSpec() const {
    CHECK(category_ == Character);
    return std::get<CharacterTypeSpec>(typeSpec_);
  }
  const DerivedTypeSpec &derivedTypeSpec() const {
    CHECK(category_ == TypeDerived || category_ == ClassDerived);
    return std::get<DerivedTypeSpec>(typeSpec_);
  }
  DerivedTypeSpec &derivedTypeSpec() {
    CHECK(category_ == TypeDerived || category_ == ClassDerived);
    return std::get<DerivedTypeSpec>(typeSpec_);
  }

  inline IntrinsicTypeSpec *AsIntrinsic();
  inline const IntrinsicTypeSpec *AsIntrinsic() const;
  inline DerivedTypeSpec *AsDerived();
  inline const DerivedTypeSpec *AsDerived() const;

  std::string AsFortran() const;

private:
  Category category_;
  std::variant<std::monostate, NumericTypeSpec, LogicalTypeSpec,
      CharacterTypeSpec, DerivedTypeSpec>
      typeSpec_;
};
llvm::raw_ostream &operator<<(llvm::raw_ostream &, const DeclTypeSpec &);

// Define some member functions here in the header so that they can be used by
// lib/Evaluate without link-time dependency on Semantics.

inline bool ArraySpec::IsExplicitShape() const {
  return CheckAll([](const ShapeSpec &x) { return x.ubound().isExplicit(); });
}
inline bool ArraySpec::CanBeAssumedShape() const {
  return CheckAll([](const ShapeSpec &x) { return x.ubound().isColon(); });
}
inline bool ArraySpec::CanBeDeferredShape() const {
  return CheckAll([](const ShapeSpec &x) {
    return x.lbound().isColon() && x.ubound().isColon();
  });
}
inline bool ArraySpec::CanBeImpliedShape() const {
  return !IsAssumedRank() &&
      CheckAll([](const ShapeSpec &x) { return x.ubound().isStar(); });
}
inline bool ArraySpec::CanBeAssumedSize() const {
  return !empty() && !IsAssumedRank() && back().ubound().isStar() &&
      std::all_of(begin(), end() - 1,
          [](const ShapeSpec &x) { return x.ubound().isExplicit(); });
}
inline bool ArraySpec::IsAssumedRank() const {
  return Rank() == 1 && front().lbound().isStar();
}

inline IntrinsicTypeSpec *DeclTypeSpec::AsIntrinsic() {
  switch (category_) {
  case Numeric:
    return &std::get<NumericTypeSpec>(typeSpec_);
  case Logical:
    return &std::get<LogicalTypeSpec>(typeSpec_);
  case Character:
    return &std::get<CharacterTypeSpec>(typeSpec_);
  default:
    return nullptr;
  }
}
inline const IntrinsicTypeSpec *DeclTypeSpec::AsIntrinsic() const {
  return const_cast<DeclTypeSpec *>(this)->AsIntrinsic();
}

inline DerivedTypeSpec *DeclTypeSpec::AsDerived() {
  switch (category_) {
  case TypeDerived:
  case ClassDerived:
    return &std::get<DerivedTypeSpec>(typeSpec_);
  default:
    return nullptr;
  }
}
inline const DerivedTypeSpec *DeclTypeSpec::AsDerived() const {
  return const_cast<DeclTypeSpec *>(this)->AsDerived();
}

bool IsInteroperableIntrinsicType(
    const DeclTypeSpec &, const common::LanguageFeatureControl &);

} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_TYPE_H_
