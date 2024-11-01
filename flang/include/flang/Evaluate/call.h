//===-- include/flang/Evaluate/call.h ---------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_EVALUATE_CALL_H_
#define FORTRAN_EVALUATE_CALL_H_

#include "common.h"
#include "constant.h"
#include "formatting.h"
#include "type.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/indirection.h"
#include "flang/Common/reference.h"
#include "flang/Parser/char-block.h"
#include "flang/Semantics/attr.h"
#include <optional>
#include <vector>

namespace llvm {
class raw_ostream;
}

namespace Fortran::semantics {
class Symbol;
}

// Mutually referential data structures are represented here with forward
// declarations of hitherto undefined class types and a level of indirection.
namespace Fortran::evaluate {
class Component;
class IntrinsicProcTable;
} // namespace Fortran::evaluate
namespace Fortran::evaluate::characteristics {
struct DummyArgument;
struct Procedure;
} // namespace Fortran::evaluate::characteristics

extern template class Fortran::common::Indirection<Fortran::evaluate::Component,
    true>;
extern template class Fortran::common::Indirection<
    Fortran::evaluate::characteristics::Procedure, true>;

namespace Fortran::evaluate {

using semantics::Symbol;
using SymbolRef = common::Reference<const Symbol>;

class ActualArgument {
public:
  // Dummy arguments that are TYPE(*) can be forwarded as actual arguments.
  // Since that's the only thing one may do with them in Fortran, they're
  // represented in expressions as a special case of an actual argument.
  class AssumedType {
  public:
    explicit AssumedType(const Symbol &);
    DEFAULT_CONSTRUCTORS_AND_ASSIGNMENTS(AssumedType)
    const Symbol &symbol() const { return symbol_; }
    int Rank() const;
    bool operator==(const AssumedType &that) const {
      return &*symbol_ == &*that.symbol_;
    }
    llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;

  private:
    SymbolRef symbol_;
  };

  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(ActualArgument)
  explicit ActualArgument(Expr<SomeType> &&);
  explicit ActualArgument(common::CopyableIndirection<Expr<SomeType>> &&);
  explicit ActualArgument(AssumedType);
  explicit ActualArgument(common::Label);
  ~ActualArgument();
  ActualArgument &operator=(Expr<SomeType> &&);

  Expr<SomeType> *UnwrapExpr() {
    if (auto *p{
            std::get_if<common::CopyableIndirection<Expr<SomeType>>>(&u_)}) {
      return &p->value();
    } else {
      return nullptr;
    }
  }
  const Expr<SomeType> *UnwrapExpr() const {
    if (const auto *p{
            std::get_if<common::CopyableIndirection<Expr<SomeType>>>(&u_)}) {
      return &p->value();
    } else {
      return nullptr;
    }
  }

  const Symbol *GetAssumedTypeDummy() const {
    if (const AssumedType * aType{std::get_if<AssumedType>(&u_)}) {
      return &aType->symbol();
    } else {
      return nullptr;
    }
  }

  common::Label GetLabel() const { return std::get<common::Label>(u_); }

  std::optional<DynamicType> GetType() const;
  int Rank() const;
  bool operator==(const ActualArgument &) const;
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;

  std::optional<parser::CharBlock> keyword() const { return keyword_; }
  ActualArgument &set_keyword(parser::CharBlock x) {
    keyword_ = x;
    return *this;
  }
  bool isAlternateReturn() const {
    return std::holds_alternative<common::Label>(u_);
  }
  bool isPassedObject() const { return isPassedObject_; }
  ActualArgument &set_isPassedObject(bool yes = true) {
    isPassedObject_ = yes;
    return *this;
  }

  bool Matches(const characteristics::DummyArgument &) const;
  common::Intent dummyIntent() const { return dummyIntent_; }
  ActualArgument &set_dummyIntent(common::Intent intent) {
    dummyIntent_ = intent;
    return *this;
  }
  std::optional<parser::CharBlock> sourceLocation() const {
    return sourceLocation_;
  }
  ActualArgument &set_sourceLocation(std::optional<parser::CharBlock> at) {
    sourceLocation_ = at;
    return *this;
  }

  // Wrap this argument in parentheses
  void Parenthesize();

  // TODO: Mark legacy %VAL and %REF arguments

private:
  // Subtlety: There is a distinction that must be maintained here between an
  // actual argument expression that is a variable and one that is not,
  // e.g. between X and (X).  The parser attempts to parse each argument
  // first as a variable, then as an expression, and the distinction appears
  // in the parse tree.
  std::variant<common::CopyableIndirection<Expr<SomeType>>, AssumedType,
      common::Label>
      u_;
  std::optional<parser::CharBlock> keyword_;
  bool isPassedObject_{false};
  common::Intent dummyIntent_{common::Intent::Default};
  std::optional<parser::CharBlock> sourceLocation_;
};

using ActualArguments = std::vector<std::optional<ActualArgument>>;

// Intrinsics are identified by their names and the characteristics
// of their arguments, at least for now.
using IntrinsicProcedure = std::string;

struct SpecificIntrinsic {
  SpecificIntrinsic(IntrinsicProcedure, characteristics::Procedure &&);
  DECLARE_CONSTRUCTORS_AND_ASSIGNMENTS(SpecificIntrinsic)
  ~SpecificIntrinsic();
  bool operator==(const SpecificIntrinsic &) const;
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;

  IntrinsicProcedure name;
  bool isRestrictedSpecific{false}; // if true, can only call it, not pass it
  common::CopyableIndirection<characteristics::Procedure> characteristics;
};

struct ProcedureDesignator {
  EVALUATE_UNION_CLASS_BOILERPLATE(ProcedureDesignator)
  explicit ProcedureDesignator(SpecificIntrinsic &&i) : u{std::move(i)} {}
  explicit ProcedureDesignator(const Symbol &n) : u{n} {}
  explicit ProcedureDesignator(Component &&);

  // Exactly one of these will return a non-null pointer.
  const SpecificIntrinsic *GetSpecificIntrinsic() const;
  const Symbol *GetSymbol() const; // symbol or component symbol
  const SymbolRef *UnwrapSymbolRef() const; // null if intrinsic or component

  // For references to NOPASS components and bindings only.
  // References to PASS components and bindings are represented
  // with the symbol below and the base object DataRef in the
  // passed-object ActualArgument.
  // Always null when the procedure is intrinsic.
  const Component *GetComponent() const;

  const Symbol *GetInterfaceSymbol() const;

  std::string GetName() const;
  std::optional<DynamicType> GetType() const;
  int Rank() const;
  bool IsElemental() const;
  bool IsPure() const;
  std::optional<Expr<SubscriptInteger>> LEN() const;
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;

  std::variant<SpecificIntrinsic, SymbolRef,
      common::CopyableIndirection<Component>>
      u;
};

using Chevrons = std::vector<Expr<SomeType>>;

class ProcedureRef {
public:
  CLASS_BOILERPLATE(ProcedureRef)
  ProcedureRef(ProcedureDesignator &&p, ActualArguments &&a,
      bool hasAlternateReturns = false)
      : proc_{std::move(p)}, arguments_{std::move(a)},
        hasAlternateReturns_{hasAlternateReturns} {}
  ~ProcedureRef();
  static void Deleter(ProcedureRef *);

  ProcedureDesignator &proc() { return proc_; }
  const ProcedureDesignator &proc() const { return proc_; }
  ActualArguments &arguments() { return arguments_; }
  const ActualArguments &arguments() const { return arguments_; }
  // CALL subr <<< kernel launch >>> (...); not function
  Chevrons &chevrons() { return chevrons_; }
  const Chevrons &chevrons() const { return chevrons_; }
  void set_chevrons(Chevrons &&chevrons) { chevrons_ = std::move(chevrons); }

  std::optional<Expr<SubscriptInteger>> LEN() const;
  int Rank() const;
  bool IsElemental() const { return proc_.IsElemental(); }
  bool hasAlternateReturns() const { return hasAlternateReturns_; }

  Expr<SomeType> *UnwrapArgExpr(int n) {
    if (static_cast<std::size_t>(n) < arguments_.size() && arguments_[n]) {
      return arguments_[n]->UnwrapExpr();
    } else {
      return nullptr;
    }
  }
  const Expr<SomeType> *UnwrapArgExpr(int n) const {
    if (static_cast<std::size_t>(n) < arguments_.size() && arguments_[n]) {
      return arguments_[n]->UnwrapExpr();
    } else {
      return nullptr;
    }
  }

  bool operator==(const ProcedureRef &) const;
  llvm::raw_ostream &AsFortran(llvm::raw_ostream &) const;

protected:
  ProcedureDesignator proc_;
  ActualArguments arguments_;
  Chevrons chevrons_;
  bool hasAlternateReturns_;
};

template <typename A> class FunctionRef : public ProcedureRef {
public:
  using Result = A;
  CLASS_BOILERPLATE(FunctionRef)
  explicit FunctionRef(ProcedureRef &&pr) : ProcedureRef{std::move(pr)} {}
  FunctionRef(ProcedureDesignator &&p, ActualArguments &&a)
      : ProcedureRef{std::move(p), std::move(a)} {}

  std::optional<DynamicType> GetType() const { return proc_.GetType(); }
};
} // namespace Fortran::evaluate
#endif // FORTRAN_EVALUATE_CALL_H_
