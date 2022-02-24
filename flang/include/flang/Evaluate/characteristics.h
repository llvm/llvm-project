//===-- include/flang/Evaluate/characteristics.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Defines data structures to represent "characteristics" of Fortran
// procedures and other entities as they are specified in section 15.3
// of Fortran 2018.

#ifndef FORTRAN_EVALUATE_CHARACTERISTICS_H_
#define FORTRAN_EVALUATE_CHARACTERISTICS_H_

#include "common.h"
#include "expression.h"
#include "shape.h"
#include "type.h"
#include "flang/Common/Fortran-features.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/enum-set.h"
#include "flang/Common/idioms.h"
#include "flang/Common/indirection.h"
#include "flang/Parser/char-block.h"
#include "flang/Semantics/symbol.h"
#include <optional>
#include <string>
#include <variant>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace Fortran::evaluate::characteristics {
struct Procedure;
}
extern template class Fortran::common::Indirection<
    Fortran::evaluate::characteristics::Procedure, true>;

namespace Fortran::evaluate::characteristics {

using common::CopyableIndirection;

// Are these procedures distinguishable for a generic name or FINAL?
bool Distinguishable(const common::LanguageFeatureControl &, const Procedure &,
    const Procedure &);
// Are these procedures distinguishable for a generic operator or assignment?
bool DistinguishableOpOrAssign(const common::LanguageFeatureControl &,
    const Procedure &, const Procedure &);

// Shapes of function results and dummy arguments have to have
// the same rank, the same deferred dimensions, and the same
// values for explicit dimensions when constant.
bool ShapesAreCompatible(const Shape &, const Shape &);

class TypeAndShape {
public:
  ENUM_CLASS(
      Attr, AssumedRank, AssumedShape, AssumedSize, DeferredShape, Coarray)
  using Attrs = common::EnumSet<Attr, Attr_enumSize>;

  explicit TypeAndShape(DynamicType t) : type_{t} { AcquireLEN(); }
  TypeAndShape(DynamicType t, int rank) : type_{t}, shape_(rank) {
    AcquireLEN();
  }
  TypeAndShape(DynamicType t, Shape &&s) : type_{t}, shape_{std::move(s)} {
    AcquireLEN();
  }
  TypeAndShape(DynamicType t, std::optional<Shape> &&s) : type_{t} {
    if (s) {
      shape_ = std::move(*s);
    }
    AcquireLEN();
  }
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(TypeAndShape)

  bool operator==(const TypeAndShape &) const;
  bool operator!=(const TypeAndShape &that) const { return !(*this == that); }

  static std::optional<TypeAndShape> Characterize(
      const semantics::Symbol &, FoldingContext &);
  static std::optional<TypeAndShape> Characterize(
      const semantics::ProcInterface &, FoldingContext &);
  static std::optional<TypeAndShape> Characterize(
      const semantics::DeclTypeSpec &, FoldingContext &);
  static std::optional<TypeAndShape> Characterize(
      const ActualArgument &, FoldingContext &);

  // Handle Expr<T> & Designator<T>
  template <typename A>
  static std::optional<TypeAndShape> Characterize(
      const A &x, FoldingContext &context) {
    if (const auto *symbol{UnwrapWholeSymbolOrComponentDataRef(x)}) {
      if (auto result{Characterize(*symbol, context)}) {
        return result;
      }
    }
    if (auto type{x.GetType()}) {
      TypeAndShape result{*type, GetShape(context, x)};
      if (type->category() == TypeCategory::Character) {
        if (const auto *chExpr{UnwrapExpr<Expr<SomeCharacter>>(x)}) {
          if (auto length{chExpr->LEN()}) {
            result.set_LEN(std::move(*length));
          }
        }
      }
      return std::move(result.Rewrite(context));
    }
    return std::nullopt;
  }

  template <typename A>
  static std::optional<TypeAndShape> Characterize(
      const std::optional<A> &x, FoldingContext &context) {
    if (x) {
      return Characterize(*x, context);
    } else {
      return std::nullopt;
    }
  }
  template <typename A>
  static std::optional<TypeAndShape> Characterize(
      const A *p, FoldingContext &context) {
    if (p) {
      return Characterize(*p, context);
    } else {
      return std::nullopt;
    }
  }

  DynamicType type() const { return type_; }
  TypeAndShape &set_type(DynamicType t) {
    type_ = t;
    return *this;
  }
  const std::optional<Expr<SubscriptInteger>> &LEN() const { return LEN_; }
  TypeAndShape &set_LEN(Expr<SubscriptInteger> &&len) {
    LEN_ = std::move(len);
    return *this;
  }
  const Shape &shape() const { return shape_; }
  const Attrs &attrs() const { return attrs_; }
  int corank() const { return corank_; }

  int Rank() const { return GetRank(shape_); }
  bool IsCompatibleWith(parser::ContextualMessages &, const TypeAndShape &that,
      const char *thisIs = "pointer", const char *thatIs = "target",
      bool omitShapeConformanceCheck = false,
      enum CheckConformanceFlags::Flags = CheckConformanceFlags::None) const;
  std::optional<Expr<SubscriptInteger>> MeasureElementSizeInBytes(
      FoldingContext &, bool align) const;
  std::optional<Expr<SubscriptInteger>> MeasureSizeInBytes(
      FoldingContext &) const;

  // called by Fold() to rewrite in place
  TypeAndShape &Rewrite(FoldingContext &);

  std::string AsFortran() const;
  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;

private:
  static std::optional<TypeAndShape> Characterize(
      const semantics::AssocEntityDetails &, FoldingContext &);
  static std::optional<TypeAndShape> Characterize(
      const semantics::ProcEntityDetails &, FoldingContext &);
  void AcquireAttrs(const semantics::Symbol &);
  void AcquireLEN();
  void AcquireLEN(const semantics::Symbol &);

protected:
  DynamicType type_;
  std::optional<Expr<SubscriptInteger>> LEN_;
  Shape shape_;
  Attrs attrs_;
  int corank_{0};
};

// 15.3.2.2
struct DummyDataObject {
  ENUM_CLASS(Attr, Optional, Allocatable, Asynchronous, Contiguous, Value,
      Volatile, Pointer, Target)
  using Attrs = common::EnumSet<Attr, Attr_enumSize>;
  DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(DummyDataObject)
  explicit DummyDataObject(const TypeAndShape &t) : type{t} {}
  explicit DummyDataObject(TypeAndShape &&t) : type{std::move(t)} {}
  explicit DummyDataObject(DynamicType t) : type{t} {}
  bool operator==(const DummyDataObject &) const;
  bool operator!=(const DummyDataObject &that) const {
    return !(*this == that);
  }
  bool IsCompatibleWith(const DummyDataObject &) const;
  static std::optional<DummyDataObject> Characterize(
      const semantics::Symbol &, FoldingContext &);
  bool CanBePassedViaImplicitInterface() const;
  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;
  TypeAndShape type;
  std::vector<Expr<SubscriptInteger>> coshape;
  common::Intent intent{common::Intent::Default};
  Attrs attrs;
};

// 15.3.2.3
struct DummyProcedure {
  ENUM_CLASS(Attr, Pointer, Optional)
  using Attrs = common::EnumSet<Attr, Attr_enumSize>;
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(DummyProcedure)
  explicit DummyProcedure(Procedure &&);
  bool operator==(const DummyProcedure &) const;
  bool operator!=(const DummyProcedure &that) const { return !(*this == that); }
  bool IsCompatibleWith(const DummyProcedure &) const;
  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;

  CopyableIndirection<Procedure> procedure;
  common::Intent intent{common::Intent::Default};
  Attrs attrs;
};

// 15.3.2.4
struct AlternateReturn {
  bool operator==(const AlternateReturn &) const { return true; }
  bool operator!=(const AlternateReturn &) const { return false; }
  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;
};

// 15.3.2.1
struct DummyArgument {
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(DummyArgument)
  DummyArgument(std::string &&name, DummyDataObject &&x)
      : name{std::move(name)}, u{std::move(x)} {}
  DummyArgument(std::string &&name, DummyProcedure &&x)
      : name{std::move(name)}, u{std::move(x)} {}
  explicit DummyArgument(AlternateReturn &&x) : u{std::move(x)} {}
  ~DummyArgument();
  bool operator==(const DummyArgument &) const;
  bool operator!=(const DummyArgument &that) const { return !(*this == that); }
  static std::optional<DummyArgument> FromActual(
      std::string &&, const Expr<SomeType> &, FoldingContext &);
  bool IsOptional() const;
  void SetOptional(bool = true);
  common::Intent GetIntent() const;
  void SetIntent(common::Intent);
  bool CanBePassedViaImplicitInterface() const;
  bool IsTypelessIntrinsicDummy() const;
  bool IsCompatibleWith(const DummyArgument &) const;
  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;

  // name and pass are not characteristics and so do not participate in
  // compatibility checks, but they are needed to determine whether
  // procedures are distinguishable
  std::string name;
  bool pass{false}; // is this the PASS argument of its procedure
  std::variant<DummyDataObject, DummyProcedure, AlternateReturn> u;
};

using DummyArguments = std::vector<DummyArgument>;

// 15.3.3
struct FunctionResult {
  ENUM_CLASS(Attr, Allocatable, Pointer, Contiguous)
  using Attrs = common::EnumSet<Attr, Attr_enumSize>;
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(FunctionResult)
  explicit FunctionResult(DynamicType);
  explicit FunctionResult(TypeAndShape &&);
  explicit FunctionResult(Procedure &&);
  ~FunctionResult();
  bool operator==(const FunctionResult &) const;
  bool operator!=(const FunctionResult &that) const { return !(*this == that); }
  static std::optional<FunctionResult> Characterize(
      const Symbol &, FoldingContext &);

  bool IsAssumedLengthCharacter() const;

  const Procedure *IsProcedurePointer() const {
    if (const auto *pp{std::get_if<CopyableIndirection<Procedure>>(&u)}) {
      return &pp->value();
    } else {
      return nullptr;
    }
  }
  const TypeAndShape *GetTypeAndShape() const {
    return std::get_if<TypeAndShape>(&u);
  }
  void SetType(DynamicType t) { std::get<TypeAndShape>(u).set_type(t); }
  bool CanBeReturnedViaImplicitInterface() const;
  bool IsCompatibleWith(const FunctionResult &) const;

  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;

  Attrs attrs;
  std::variant<TypeAndShape, CopyableIndirection<Procedure>> u;
};

// 15.3.1
struct Procedure {
  ENUM_CLASS(
      Attr, Pure, Elemental, BindC, ImplicitInterface, NullPointer, Subroutine)
  using Attrs = common::EnumSet<Attr, Attr_enumSize>;
  Procedure(){};
  Procedure(FunctionResult &&, DummyArguments &&, Attrs);
  Procedure(DummyArguments &&, Attrs); // for subroutines and NULL()
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(Procedure)
  ~Procedure();
  bool operator==(const Procedure &) const;
  bool operator!=(const Procedure &that) const { return !(*this == that); }

  // Characterizes a procedure.  If a Symbol, it may be an
  // "unrestricted specific intrinsic function".
  // Error messages are produced when a procedure cannot be characterized.
  static std::optional<Procedure> Characterize(
      const semantics::Symbol &, FoldingContext &);
  static std::optional<Procedure> Characterize(
      const ProcedureDesignator &, FoldingContext &);
  static std::optional<Procedure> Characterize(
      const ProcedureRef &, FoldingContext &);

  // At most one of these will return true.
  // For "EXTERNAL P" with no type for or calls to P, both will be false.
  bool IsFunction() const { return functionResult.has_value(); }
  bool IsSubroutine() const { return attrs.test(Attr::Subroutine); }

  bool IsPure() const { return attrs.test(Attr::Pure); }
  bool IsElemental() const { return attrs.test(Attr::Elemental); }
  bool IsBindC() const { return attrs.test(Attr::BindC); }
  bool HasExplicitInterface() const {
    return !attrs.test(Attr::ImplicitInterface);
  }
  int FindPassIndex(std::optional<parser::CharBlock>) const;
  bool CanBeCalledViaImplicitInterface() const;
  bool CanOverride(const Procedure &, std::optional<int> passIndex) const;
  bool IsCompatibleWith(const Procedure &) const;

  llvm::raw_ostream &Dump(llvm::raw_ostream &) const;

  std::optional<FunctionResult> functionResult;
  DummyArguments dummyArguments;
  Attrs attrs;
};
} // namespace Fortran::evaluate::characteristics
#endif // FORTRAN_EVALUATE_CHARACTERISTICS_H_
