//===-- lib/Semantics/check-declarations.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Static declaration checking

#include "check-declarations.h"
#include "pointer-assignment.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include <algorithm>
#include <map>
#include <string>

namespace Fortran::semantics {

namespace characteristics = evaluate::characteristics;
using characteristics::DummyArgument;
using characteristics::DummyDataObject;
using characteristics::DummyProcedure;
using characteristics::FunctionResult;
using characteristics::Procedure;

class CheckHelper {
public:
  explicit CheckHelper(SemanticsContext &c) : context_{c} {}

  SemanticsContext &context() { return context_; }
  void Check() { Check(context_.globalScope()); }
  void Check(const ParamValue &, bool canBeAssumed);
  void Check(const Bound &bound) { CheckSpecExpr(bound.GetExplicit()); }
  void Check(const ShapeSpec &spec) {
    Check(spec.lbound());
    Check(spec.ubound());
  }
  void Check(const ArraySpec &);
  void Check(const DeclTypeSpec &, bool canHaveAssumedTypeParameters);
  void Check(const Symbol &);
  void Check(const Scope &);
  const Procedure *Characterize(const Symbol &);

private:
  template <typename A> void CheckSpecExpr(const A &x) {
    evaluate::CheckSpecificationExpr(x, DEREF(scope_), foldingContext_);
  }
  void CheckValue(const Symbol &, const DerivedTypeSpec *);
  void CheckVolatile(const Symbol &, const DerivedTypeSpec *);
  void CheckPointer(const Symbol &);
  void CheckPassArg(
      const Symbol &proc, const Symbol *interface, const WithPassArg &);
  void CheckProcBinding(const Symbol &, const ProcBindingDetails &);
  void CheckObjectEntity(const Symbol &, const ObjectEntityDetails &);
  void CheckPointerInitialization(const Symbol &);
  void CheckArraySpec(const Symbol &, const ArraySpec &);
  void CheckProcEntity(const Symbol &, const ProcEntityDetails &);
  void CheckSubprogram(const Symbol &, const SubprogramDetails &);
  void CheckAssumedTypeEntity(const Symbol &, const ObjectEntityDetails &);
  void CheckDerivedType(const Symbol &, const DerivedTypeDetails &);
  bool CheckFinal(
      const Symbol &subroutine, SourceName, const Symbol &derivedType);
  bool CheckDistinguishableFinals(const Symbol &f1, SourceName f1name,
      const Symbol &f2, SourceName f2name, const Symbol &derivedType);
  void CheckGeneric(const Symbol &, const GenericDetails &);
  void CheckHostAssoc(const Symbol &, const HostAssocDetails &);
  bool CheckDefinedOperator(
      SourceName, GenericKind, const Symbol &, const Procedure &);
  std::optional<parser::MessageFixedText> CheckNumberOfArgs(
      const GenericKind &, std::size_t);
  bool CheckDefinedOperatorArg(
      const SourceName &, const Symbol &, const Procedure &, std::size_t);
  bool CheckDefinedAssignment(const Symbol &, const Procedure &);
  bool CheckDefinedAssignmentArg(const Symbol &, const DummyArgument &, int);
  void CheckSpecificsAreDistinguishable(const Symbol &, const GenericDetails &);
  void CheckEquivalenceSet(const EquivalenceSet &);
  void CheckBlockData(const Scope &);
  void CheckGenericOps(const Scope &);
  bool CheckConflicting(const Symbol &, Attr, Attr);
  void WarnMissingFinal(const Symbol &);
  bool InPure() const {
    return innermostSymbol_ && IsPureProcedure(*innermostSymbol_);
  }
  bool InElemental() const {
    return innermostSymbol_ && innermostSymbol_->attrs().test(Attr::ELEMENTAL);
  }
  bool InFunction() const {
    return innermostSymbol_ && IsFunction(*innermostSymbol_);
  }
  template <typename... A>
  void SayWithDeclaration(const Symbol &symbol, A &&...x) {
    if (parser::Message * msg{messages_.Say(std::forward<A>(x)...)}) {
      if (messages_.at().begin() != symbol.name().begin()) {
        evaluate::AttachDeclaration(*msg, symbol);
      }
    }
  }
  bool IsResultOkToDiffer(const FunctionResult &);
  void CheckBindCName(const Symbol &);
  // Check functions for defined I/O procedures
  void CheckDefinedIoProc(
      const Symbol &, const GenericDetails &, GenericKind::DefinedIo);
  bool CheckDioDummyIsData(const Symbol &, const Symbol *, std::size_t);
  void CheckDioDummyIsDerived(
      const Symbol &, const Symbol &, GenericKind::DefinedIo ioKind);
  void CheckDioDummyIsDefaultInteger(const Symbol &, const Symbol &);
  void CheckDioDummyIsScalar(const Symbol &, const Symbol &);
  void CheckDioDummyAttrs(const Symbol &, const Symbol &, Attr);
  void CheckDioDtvArg(const Symbol &, const Symbol *, GenericKind::DefinedIo);
  void CheckGenericVsIntrinsic(const Symbol &, const GenericDetails &);
  void CheckDefaultIntegerArg(const Symbol &, const Symbol *, Attr);
  void CheckDioAssumedLenCharacterArg(
      const Symbol &, const Symbol *, std::size_t, Attr);
  void CheckDioVlistArg(const Symbol &, const Symbol *, std::size_t);
  void CheckDioArgCount(
      const Symbol &, GenericKind::DefinedIo ioKind, std::size_t);
  struct TypeWithDefinedIo {
    const DerivedTypeSpec *type;
    GenericKind::DefinedIo ioKind;
    const Symbol &proc;
  };
  void CheckAlreadySeenDefinedIo(
      const DerivedTypeSpec *, GenericKind::DefinedIo, const Symbol &);

  SemanticsContext &context_;
  evaluate::FoldingContext &foldingContext_{context_.foldingContext()};
  parser::ContextualMessages &messages_{foldingContext_.messages()};
  const Scope *scope_{nullptr};
  bool scopeIsUninstantiatedPDT_{false};
  // This symbol is the one attached to the innermost enclosing scope
  // that has a symbol.
  const Symbol *innermostSymbol_{nullptr};
  // Cache of calls to Procedure::Characterize(Symbol)
  std::map<SymbolRef, std::optional<Procedure>, SymbolAddressCompare>
      characterizeCache_;
  // Collection of symbols with BIND(C) names
  std::map<std::string, SymbolRef> bindC_;
  // Derived types that have defined input/output procedures
  std::vector<TypeWithDefinedIo> seenDefinedIoTypes_;
};

class DistinguishabilityHelper {
public:
  DistinguishabilityHelper(SemanticsContext &context) : context_{context} {}
  void Add(const Symbol &, GenericKind, const Symbol &, const Procedure &);
  void Check(const Scope &);

private:
  void SayNotDistinguishable(const Scope &, const SourceName &, GenericKind,
      const Symbol &, const Symbol &);
  void AttachDeclaration(parser::Message &, const Scope &, const Symbol &);

  SemanticsContext &context_;
  struct ProcedureInfo {
    GenericKind kind;
    const Symbol &symbol;
    const Procedure &procedure;
  };
  std::map<SourceName, std::vector<ProcedureInfo>> nameToInfo_;
};

void CheckHelper::Check(const ParamValue &value, bool canBeAssumed) {
  if (value.isAssumed()) {
    if (!canBeAssumed) { // C795, C721, C726
      messages_.Say(
          "An assumed (*) type parameter may be used only for a (non-statement"
          " function) dummy argument, associate name, named constant, or"
          " external function result"_err_en_US);
    }
  } else {
    CheckSpecExpr(value.GetExplicit());
  }
}

void CheckHelper::Check(const ArraySpec &shape) {
  for (const auto &spec : shape) {
    Check(spec);
  }
}

void CheckHelper::Check(
    const DeclTypeSpec &type, bool canHaveAssumedTypeParameters) {
  if (type.category() == DeclTypeSpec::Character) {
    Check(type.characterTypeSpec().length(), canHaveAssumedTypeParameters);
  } else if (const DerivedTypeSpec * derived{type.AsDerived()}) {
    for (auto &parm : derived->parameters()) {
      Check(parm.second, canHaveAssumedTypeParameters);
    }
  }
}

void CheckHelper::Check(const Symbol &symbol) {
  if (symbol.name().size() > common::maxNameLen) {
    messages_.Say(symbol.name(),
        "%s has length %d, which is greater than the maximum name length "
        "%d"_port_en_US,
        symbol.name(), symbol.name().size(), common::maxNameLen);
  }
  if (context_.HasError(symbol)) {
    return;
  }
  auto restorer{messages_.SetLocation(symbol.name())};
  context_.set_location(symbol.name());
  const DeclTypeSpec *type{symbol.GetType()};
  const DerivedTypeSpec *derived{type ? type->AsDerived() : nullptr};
  bool isDone{false};
  common::visit(
      common::visitors{
          [&](const UseDetails &x) { isDone = true; },
          [&](const HostAssocDetails &x) {
            CheckHostAssoc(symbol, x);
            isDone = true;
          },
          [&](const ProcBindingDetails &x) {
            CheckProcBinding(symbol, x);
            isDone = true;
          },
          [&](const ObjectEntityDetails &x) { CheckObjectEntity(symbol, x); },
          [&](const ProcEntityDetails &x) { CheckProcEntity(symbol, x); },
          [&](const SubprogramDetails &x) { CheckSubprogram(symbol, x); },
          [&](const DerivedTypeDetails &x) { CheckDerivedType(symbol, x); },
          [&](const GenericDetails &x) { CheckGeneric(symbol, x); },
          [](const auto &) {},
      },
      symbol.details());
  if (symbol.attrs().test(Attr::VOLATILE)) {
    CheckVolatile(symbol, derived);
  }
  CheckBindCName(symbol);
  if (isDone) {
    return; // following checks do not apply
  }
  if (IsPointer(symbol)) {
    CheckPointer(symbol);
  }
  if (InPure()) {
    if (IsSaved(symbol)) {
      if (IsInitialized(symbol)) {
        messages_.Say(
            "A pure subprogram may not initialize a variable"_err_en_US);
      } else {
        messages_.Say(
            "A pure subprogram may not have a variable with the SAVE attribute"_err_en_US);
      }
    }
    if (symbol.attrs().test(Attr::VOLATILE)) {
      messages_.Say(
          "A pure subprogram may not have a variable with the VOLATILE attribute"_err_en_US);
    }
    if (IsProcedure(symbol) && !IsPureProcedure(symbol) && IsDummy(symbol)) {
      messages_.Say(
          "A dummy procedure of a pure subprogram must be pure"_err_en_US);
    }
    if (!IsDummy(symbol) && !IsFunctionResult(symbol)) {
      if (IsPolymorphicAllocatable(symbol)) {
        SayWithDeclaration(symbol,
            "Deallocation of polymorphic object '%s' is not permitted in a pure subprogram"_err_en_US,
            symbol.name());
      } else if (derived) {
        if (auto bad{FindPolymorphicAllocatableUltimateComponent(*derived)}) {
          SayWithDeclaration(*bad,
              "Deallocation of polymorphic object '%s%s' is not permitted in a pure subprogram"_err_en_US,
              symbol.name(), bad.BuildResultDesignatorName());
        }
      }
    }
  }
  if (type) { // Section 7.2, paragraph 7
    bool canHaveAssumedParameter{IsNamedConstant(symbol) ||
        (IsAssumedLengthCharacter(symbol) && // C722
            IsExternal(symbol)) ||
        symbol.test(Symbol::Flag::ParentComp)};
    if (!IsStmtFunctionDummy(symbol)) { // C726
      if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
        canHaveAssumedParameter |= object->isDummy() ||
            (object->isFuncResult() &&
                type->category() == DeclTypeSpec::Character) ||
            IsStmtFunctionResult(symbol); // Avoids multiple messages
      } else {
        canHaveAssumedParameter |= symbol.has<AssocEntityDetails>();
      }
    }
    Check(*type, canHaveAssumedParameter);
    if (InPure() && InFunction() && IsFunctionResult(symbol)) {
      if (derived && HasImpureFinal(*derived)) { // C1584
        messages_.Say(
            "Result of pure function may not have an impure FINAL subroutine"_err_en_US);
      }
      if (type->IsPolymorphic() && IsAllocatable(symbol)) { // C1585
        messages_.Say(
            "Result of pure function may not be both polymorphic and ALLOCATABLE"_err_en_US);
      }
      if (derived) {
        if (auto bad{FindPolymorphicAllocatableUltimateComponent(*derived)}) {
          SayWithDeclaration(*bad,
              "Result of pure function may not have polymorphic ALLOCATABLE ultimate component '%s'"_err_en_US,
              bad.BuildResultDesignatorName());
        }
      }
    }
  }
  if (IsAssumedLengthCharacter(symbol) && IsExternal(symbol)) { // C723
    if (symbol.attrs().test(Attr::RECURSIVE)) {
      messages_.Say(
          "An assumed-length CHARACTER(*) function cannot be RECURSIVE"_err_en_US);
    }
    if (symbol.Rank() > 0) {
      messages_.Say(
          "An assumed-length CHARACTER(*) function cannot return an array"_err_en_US);
    }
    if (symbol.attrs().test(Attr::PURE)) {
      messages_.Say(
          "An assumed-length CHARACTER(*) function cannot be PURE"_err_en_US);
    }
    if (symbol.attrs().test(Attr::ELEMENTAL)) {
      messages_.Say(
          "An assumed-length CHARACTER(*) function cannot be ELEMENTAL"_err_en_US);
    }
    if (const Symbol * result{FindFunctionResult(symbol)}) {
      if (IsPointer(*result)) {
        messages_.Say(
            "An assumed-length CHARACTER(*) function cannot return a POINTER"_err_en_US);
      }
    }
  }
  if (symbol.attrs().test(Attr::VALUE)) {
    CheckValue(symbol, derived);
  }
  if (symbol.attrs().test(Attr::CONTIGUOUS) && IsPointer(symbol) &&
      symbol.Rank() == 0) { // C830
    messages_.Say("CONTIGUOUS POINTER must be an array"_err_en_US);
  }
  if (IsDummy(symbol)) {
    if (IsNamedConstant(symbol)) {
      messages_.Say(
          "A dummy argument may not also be a named constant"_err_en_US);
    }
    if (!symbol.test(Symbol::Flag::InDataStmt) /*caught elsewhere*/ &&
        IsSaved(symbol)) {
      messages_.Say(
          "A dummy argument may not have the SAVE attribute"_err_en_US);
    }
  } else if (IsFunctionResult(symbol)) {
    if (!symbol.test(Symbol::Flag::InDataStmt) /*caught elsewhere*/ &&
        IsSaved(symbol)) {
      messages_.Say(
          "A function result may not have the SAVE attribute"_err_en_US);
    }
  }
  if (symbol.owner().IsDerivedType() &&
      (symbol.attrs().test(Attr::CONTIGUOUS) &&
          !(IsPointer(symbol) && symbol.Rank() > 0))) { // C752
    messages_.Say(
        "A CONTIGUOUS component must be an array with the POINTER attribute"_err_en_US);
  }
  if (symbol.owner().IsModule() && IsAutomatic(symbol)) {
    messages_.Say(
        "Automatic data object '%s' may not appear in the specification part"
        " of a module"_err_en_US,
        symbol.name());
  }
}

void CheckHelper::CheckValue(
    const Symbol &symbol, const DerivedTypeSpec *derived) { // C863 - C865
  if (!IsDummy(symbol)) {
    messages_.Say(
        "VALUE attribute may apply only to a dummy argument"_err_en_US);
  }
  if (IsProcedure(symbol)) {
    messages_.Say(
        "VALUE attribute may apply only to a dummy data object"_err_en_US);
  }
  if (IsAssumedSizeArray(symbol)) {
    messages_.Say(
        "VALUE attribute may not apply to an assumed-size array"_err_en_US);
  }
  if (evaluate::IsCoarray(symbol)) {
    messages_.Say("VALUE attribute may not apply to a coarray"_err_en_US);
  }
  if (IsAllocatable(symbol)) {
    messages_.Say("VALUE attribute may not apply to an ALLOCATABLE"_err_en_US);
  } else if (IsPointer(symbol)) {
    messages_.Say("VALUE attribute may not apply to a POINTER"_err_en_US);
  }
  if (IsIntentInOut(symbol)) {
    messages_.Say(
        "VALUE attribute may not apply to an INTENT(IN OUT) argument"_err_en_US);
  } else if (IsIntentOut(symbol)) {
    messages_.Say(
        "VALUE attribute may not apply to an INTENT(OUT) argument"_err_en_US);
  }
  if (symbol.attrs().test(Attr::VOLATILE)) {
    messages_.Say("VALUE attribute may not apply to a VOLATILE"_err_en_US);
  }
  if (innermostSymbol_ && IsBindCProcedure(*innermostSymbol_) &&
      IsOptional(symbol)) {
    messages_.Say(
        "VALUE attribute may not apply to an OPTIONAL in a BIND(C) procedure"_err_en_US);
  }
  if (derived) {
    if (FindCoarrayUltimateComponent(*derived)) {
      messages_.Say(
          "VALUE attribute may not apply to a type with a coarray ultimate component"_err_en_US);
    }
  }
}

void CheckHelper::CheckAssumedTypeEntity( // C709
    const Symbol &symbol, const ObjectEntityDetails &details) {
  if (const DeclTypeSpec * type{symbol.GetType()};
      type && type->category() == DeclTypeSpec::TypeStar) {
    if (!IsDummy(symbol)) {
      messages_.Say(
          "Assumed-type entity '%s' must be a dummy argument"_err_en_US,
          symbol.name());
    } else {
      if (symbol.attrs().test(Attr::ALLOCATABLE)) {
        messages_.Say("Assumed-type argument '%s' cannot have the ALLOCATABLE"
                      " attribute"_err_en_US,
            symbol.name());
      }
      if (symbol.attrs().test(Attr::POINTER)) {
        messages_.Say("Assumed-type argument '%s' cannot have the POINTER"
                      " attribute"_err_en_US,
            symbol.name());
      }
      if (symbol.attrs().test(Attr::VALUE)) {
        messages_.Say("Assumed-type argument '%s' cannot have the VALUE"
                      " attribute"_err_en_US,
            symbol.name());
      }
      if (symbol.attrs().test(Attr::INTENT_OUT)) {
        messages_.Say(
            "Assumed-type argument '%s' cannot be INTENT(OUT)"_err_en_US,
            symbol.name());
      }
      if (evaluate::IsCoarray(symbol)) {
        messages_.Say(
            "Assumed-type argument '%s' cannot be a coarray"_err_en_US,
            symbol.name());
      }
      if (details.IsArray() && details.shape().IsExplicitShape()) {
        messages_.Say(
            "Assumed-type array argument 'arg8' must be assumed shape,"
            " assumed size, or assumed rank"_err_en_US,
            symbol.name());
      }
    }
  }
}

void CheckHelper::CheckObjectEntity(
    const Symbol &symbol, const ObjectEntityDetails &details) {
  CheckArraySpec(symbol, details.shape());
  Check(details.shape());
  Check(details.coshape());
  if (details.shape().Rank() > common::maxRank) {
    messages_.Say(
        "'%s' has rank %d, which is greater than the maximum supported rank %d"_err_en_US,
        symbol.name(), details.shape().Rank(), common::maxRank);
  } else if (details.shape().Rank() + details.coshape().Rank() >
      common::maxRank) {
    messages_.Say(
        "'%s' has rank %d and corank %d, whose sum is greater than the maximum supported rank %d"_err_en_US,
        symbol.name(), details.shape().Rank(), details.coshape().Rank(),
        common::maxRank);
  }
  CheckAssumedTypeEntity(symbol, details);
  WarnMissingFinal(symbol);
  if (!details.coshape().empty()) {
    bool isDeferredCoshape{details.coshape().CanBeDeferredShape()};
    if (IsAllocatable(symbol)) {
      if (!isDeferredCoshape) { // C827
        messages_.Say("'%s' is an ALLOCATABLE coarray and must have a deferred"
                      " coshape"_err_en_US,
            symbol.name());
      }
    } else if (symbol.owner().IsDerivedType()) { // C746
      std::string deferredMsg{
          isDeferredCoshape ? "" : " and have a deferred coshape"};
      messages_.Say("Component '%s' is a coarray and must have the ALLOCATABLE"
                    " attribute%s"_err_en_US,
          symbol.name(), deferredMsg);
    } else {
      if (!details.coshape().CanBeAssumedSize()) { // C828
        messages_.Say(
            "'%s' is a non-ALLOCATABLE coarray and must have an explicit coshape"_err_en_US,
            symbol.name());
      }
    }
    if (const DeclTypeSpec * type{details.type()}) {
      if (IsBadCoarrayType(type->AsDerived())) { // C747 & C824
        messages_.Say(
            "Coarray '%s' may not have type TEAM_TYPE, C_PTR, or C_FUNPTR"_err_en_US,
            symbol.name());
      }
    }
  }
  if (details.isDummy()) {
    if (symbol.attrs().test(Attr::INTENT_OUT)) {
      if (FindUltimateComponent(symbol, [](const Symbol &x) {
            return evaluate::IsCoarray(x) && IsAllocatable(x);
          })) { // C846
        messages_.Say(
            "An INTENT(OUT) dummy argument may not be, or contain, an ALLOCATABLE coarray"_err_en_US);
      }
      if (IsOrContainsEventOrLockComponent(symbol)) { // C847
        messages_.Say(
            "An INTENT(OUT) dummy argument may not be, or contain, EVENT_TYPE or LOCK_TYPE"_err_en_US);
      }
    }
    if (InPure() && !IsStmtFunction(DEREF(innermostSymbol_)) &&
        !IsPointer(symbol) && !IsIntentIn(symbol) &&
        !symbol.attrs().test(Attr::VALUE)) {
      if (InFunction()) { // C1583
        messages_.Say(
            "non-POINTER dummy argument of pure function must be INTENT(IN) or VALUE"_err_en_US);
      } else if (IsIntentOut(symbol)) {
        if (const DeclTypeSpec * type{details.type()}) {
          if (type && type->IsPolymorphic()) { // C1588
            messages_.Say(
                "An INTENT(OUT) dummy argument of a pure subroutine may not be polymorphic"_err_en_US);
          } else if (const DerivedTypeSpec * derived{type->AsDerived()}) {
            if (FindUltimateComponent(*derived, [](const Symbol &x) {
                  const DeclTypeSpec *type{x.GetType()};
                  return type && type->IsPolymorphic();
                })) { // C1588
              messages_.Say(
                  "An INTENT(OUT) dummy argument of a pure subroutine may not have a polymorphic ultimate component"_err_en_US);
            }
            if (HasImpureFinal(*derived)) { // C1587
              messages_.Say(
                  "An INTENT(OUT) dummy argument of a pure subroutine may not have an impure FINAL subroutine"_err_en_US);
            }
          }
        }
      } else if (!IsIntentInOut(symbol)) { // C1586
        messages_.Say(
            "non-POINTER dummy argument of pure subroutine must have INTENT() or VALUE attribute"_err_en_US);
      }
    }
  } else if (symbol.attrs().test(Attr::INTENT_IN) ||
      symbol.attrs().test(Attr::INTENT_OUT) ||
      symbol.attrs().test(Attr::INTENT_INOUT)) {
    messages_.Say("INTENT attributes may apply only to a dummy "
                  "argument"_err_en_US); // C843
  } else if (IsOptional(symbol)) {
    messages_.Say("OPTIONAL attribute may apply only to a dummy "
                  "argument"_err_en_US); // C849
  }
  if (InElemental()) {
    if (details.isDummy()) { // C15100
      if (details.shape().Rank() > 0) {
        messages_.Say(
            "A dummy argument of an ELEMENTAL procedure must be scalar"_err_en_US);
      }
      if (IsAllocatable(symbol)) {
        messages_.Say(
            "A dummy argument of an ELEMENTAL procedure may not be ALLOCATABLE"_err_en_US);
      }
      if (evaluate::IsCoarray(symbol)) {
        messages_.Say(
            "A dummy argument of an ELEMENTAL procedure may not be a coarray"_err_en_US);
      }
      if (IsPointer(symbol)) {
        messages_.Say(
            "A dummy argument of an ELEMENTAL procedure may not be a POINTER"_err_en_US);
      }
      if (!symbol.attrs().HasAny(Attrs{Attr::VALUE, Attr::INTENT_IN,
              Attr::INTENT_INOUT, Attr::INTENT_OUT})) { // C15102
        messages_.Say(
            "A dummy argument of an ELEMENTAL procedure must have an INTENT() or VALUE attribute"_err_en_US);
      }
    } else if (IsFunctionResult(symbol)) { // C15101
      if (details.shape().Rank() > 0) {
        messages_.Say(
            "The result of an ELEMENTAL function must be scalar"_err_en_US);
      }
      if (IsAllocatable(symbol)) {
        messages_.Say(
            "The result of an ELEMENTAL function may not be ALLOCATABLE"_err_en_US);
      }
      if (IsPointer(symbol)) {
        messages_.Say(
            "The result of an ELEMENTAL function may not be a POINTER"_err_en_US);
      }
    }
  }
  if (HasDeclarationInitializer(symbol)) { // C808; ignore DATA initialization
    CheckPointerInitialization(symbol);
    if (IsAutomatic(symbol)) {
      messages_.Say(
          "An automatic variable or component must not be initialized"_err_en_US);
    } else if (IsDummy(symbol)) {
      messages_.Say("A dummy argument must not be initialized"_err_en_US);
    } else if (IsFunctionResult(symbol)) {
      messages_.Say("A function result must not be initialized"_err_en_US);
    } else if (IsInBlankCommon(symbol)) {
      messages_.Say(
          "A variable in blank COMMON should not be initialized"_port_en_US);
    }
  }
  if (symbol.owner().kind() == Scope::Kind::BlockData) {
    if (IsAllocatable(symbol)) {
      messages_.Say(
          "An ALLOCATABLE variable may not appear in a BLOCK DATA subprogram"_err_en_US);
    } else if (IsInitialized(symbol) && !FindCommonBlockContaining(symbol)) {
      messages_.Say(
          "An initialized variable in BLOCK DATA must be in a COMMON block"_err_en_US);
    }
  }
  if (const DeclTypeSpec * type{details.type()}) { // C708
    if (type->IsPolymorphic() &&
        !(type->IsAssumedType() || IsAllocatableOrPointer(symbol) ||
            IsDummy(symbol))) {
      messages_.Say("CLASS entity '%s' must be a dummy argument or have "
                    "ALLOCATABLE or POINTER attribute"_err_en_US,
          symbol.name());
    }
  }
}

void CheckHelper::CheckPointerInitialization(const Symbol &symbol) {
  if (IsPointer(symbol) && !context_.HasError(symbol) &&
      !scopeIsUninstantiatedPDT_) {
    if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
      if (object->init()) { // C764, C765; C808
        if (auto designator{evaluate::AsGenericExpr(symbol)}) {
          auto restorer{messages_.SetLocation(symbol.name())};
          context_.set_location(symbol.name());
          CheckInitialTarget(foldingContext_, *designator, *object->init());
        }
      }
    } else if (const auto *proc{symbol.detailsIf<ProcEntityDetails>()}) {
      if (proc->init() && *proc->init()) {
        // C1519 - must be nonelemental external or module procedure,
        // or an unrestricted specific intrinsic function.
        const Symbol &ultimate{(*proc->init())->GetUltimate()};
        if (ultimate.attrs().test(Attr::INTRINSIC)) {
          if (const auto intrinsic{
                  context_.intrinsics().IsSpecificIntrinsicFunction(
                      ultimate.name().ToString())};
              !intrinsic || intrinsic->isRestrictedSpecific) { // C1030
            context_.Say(
                "Intrinsic procedure '%s' is not an unrestricted specific "
                "intrinsic permitted for use as the initializer for procedure "
                "pointer '%s'"_err_en_US,
                ultimate.name(), symbol.name());
          }
        } else if (!ultimate.attrs().test(Attr::EXTERNAL) &&
            ultimate.owner().kind() != Scope::Kind::Module) {
          context_.Say("Procedure pointer '%s' initializer '%s' is neither "
                       "an external nor a module procedure"_err_en_US,
              symbol.name(), ultimate.name());
        } else if (ultimate.attrs().test(Attr::ELEMENTAL)) {
          context_.Say("Procedure pointer '%s' cannot be initialized with the "
                       "elemental procedure '%s"_err_en_US,
              symbol.name(), ultimate.name());
        } else {
          // TODO: Check the "shalls" in the 15.4.3.6 paragraphs 7-10.
        }
      }
    }
  }
}

// The six different kinds of array-specs:
//   array-spec     -> explicit-shape-list | deferred-shape-list
//                     | assumed-shape-list | implied-shape-list
//                     | assumed-size | assumed-rank
//   explicit-shape -> [ lb : ] ub
//   deferred-shape -> :
//   assumed-shape  -> [ lb ] :
//   implied-shape  -> [ lb : ] *
//   assumed-size   -> [ explicit-shape-list , ] [ lb : ] *
//   assumed-rank   -> ..
// Note:
// - deferred-shape is also an assumed-shape
// - A single "*" or "lb:*" might be assumed-size or implied-shape-list
void CheckHelper::CheckArraySpec(
    const Symbol &symbol, const ArraySpec &arraySpec) {
  if (arraySpec.Rank() == 0) {
    return;
  }
  bool isExplicit{arraySpec.IsExplicitShape()};
  bool canBeDeferred{arraySpec.CanBeDeferredShape()};
  bool canBeImplied{arraySpec.CanBeImpliedShape()};
  bool canBeAssumedShape{arraySpec.CanBeAssumedShape()};
  bool canBeAssumedSize{arraySpec.CanBeAssumedSize()};
  bool isAssumedRank{arraySpec.IsAssumedRank()};
  std::optional<parser::MessageFixedText> msg;
  if (symbol.test(Symbol::Flag::CrayPointee) && !isExplicit &&
      !canBeAssumedSize) {
    msg = "Cray pointee '%s' must have must have explicit shape or"
          " assumed size"_err_en_US;
  } else if (IsAllocatableOrPointer(symbol) && !canBeDeferred &&
      !isAssumedRank) {
    if (symbol.owner().IsDerivedType()) { // C745
      if (IsAllocatable(symbol)) {
        msg = "Allocatable array component '%s' must have"
              " deferred shape"_err_en_US;
      } else {
        msg = "Array pointer component '%s' must have deferred shape"_err_en_US;
      }
    } else {
      if (IsAllocatable(symbol)) { // C832
        msg = "Allocatable array '%s' must have deferred shape or"
              " assumed rank"_err_en_US;
      } else {
        msg = "Array pointer '%s' must have deferred shape or"
              " assumed rank"_err_en_US;
      }
    }
  } else if (IsDummy(symbol)) {
    if (canBeImplied && !canBeAssumedSize) { // C836
      msg = "Dummy array argument '%s' may not have implied shape"_err_en_US;
    }
  } else if (canBeAssumedShape && !canBeDeferred) {
    msg = "Assumed-shape array '%s' must be a dummy argument"_err_en_US;
  } else if (canBeAssumedSize && !canBeImplied) { // C833
    msg = "Assumed-size array '%s' must be a dummy argument"_err_en_US;
  } else if (isAssumedRank) { // C837
    msg = "Assumed-rank array '%s' must be a dummy argument"_err_en_US;
  } else if (canBeImplied) {
    if (!IsNamedConstant(symbol)) { // C835, C836
      msg = "Implied-shape array '%s' must be a named constant or a "
            "dummy argument"_err_en_US;
    }
  } else if (IsNamedConstant(symbol)) {
    if (!isExplicit && !canBeImplied) {
      msg = "Named constant '%s' array must have constant or"
            " implied shape"_err_en_US;
    }
  } else if (!IsAllocatableOrPointer(symbol) && !isExplicit) {
    if (symbol.owner().IsDerivedType()) { // C749
      msg = "Component array '%s' without ALLOCATABLE or POINTER attribute must"
            " have explicit shape"_err_en_US;
    } else { // C816
      msg = "Array '%s' without ALLOCATABLE or POINTER attribute must have"
            " explicit shape"_err_en_US;
    }
  }
  if (msg) {
    context_.Say(std::move(*msg), symbol.name());
  }
}

void CheckHelper::CheckProcEntity(
    const Symbol &symbol, const ProcEntityDetails &details) {
  if (details.isDummy()) {
    if (!symbol.attrs().test(Attr::POINTER) && // C843
        (symbol.attrs().test(Attr::INTENT_IN) ||
            symbol.attrs().test(Attr::INTENT_OUT) ||
            symbol.attrs().test(Attr::INTENT_INOUT))) {
      messages_.Say("A dummy procedure without the POINTER attribute"
                    " may not have an INTENT attribute"_err_en_US);
    }
    if (InElemental()) { // C15100
      messages_.Say(
          "An ELEMENTAL subprogram may not have a dummy procedure"_err_en_US);
    }
    const Symbol *interface { details.interface().symbol() };
    if (!symbol.attrs().test(Attr::INTRINSIC) &&
        (symbol.attrs().test(Attr::ELEMENTAL) ||
            (interface && !interface->attrs().test(Attr::INTRINSIC) &&
                interface->attrs().test(Attr::ELEMENTAL)))) {
      // There's no explicit constraint or "shall" that we can find in the
      // standard for this check, but it seems to be implied in multiple
      // sites, and ELEMENTAL non-intrinsic actual arguments *are*
      // explicitly forbidden.  But we allow "PROCEDURE(SIN)::dummy"
      // because it is explicitly legal to *pass* the specific intrinsic
      // function SIN as an actual argument.
      messages_.Say("A dummy procedure may not be ELEMENTAL"_err_en_US);
    }
  } else if (symbol.attrs().test(Attr::INTENT_IN) ||
      symbol.attrs().test(Attr::INTENT_OUT) ||
      symbol.attrs().test(Attr::INTENT_INOUT)) {
    messages_.Say("INTENT attributes may apply only to a dummy "
                  "argument"_err_en_US); // C843
  } else if (IsOptional(symbol)) {
    messages_.Say("OPTIONAL attribute may apply only to a dummy "
                  "argument"_err_en_US); // C849
  } else if (symbol.owner().IsDerivedType()) {
    if (!symbol.attrs().test(Attr::POINTER)) { // C756
      const auto &name{symbol.name()};
      messages_.Say(name,
          "Procedure component '%s' must have POINTER attribute"_err_en_US,
          name);
    }
    CheckPassArg(symbol, details.interface().symbol(), details);
  }
  if (symbol.attrs().test(Attr::POINTER)) {
    CheckPointerInitialization(symbol);
    if (const Symbol * interface{details.interface().symbol()}) {
      if (interface->attrs().test(Attr::INTRINSIC)) {
        if (const auto intrinsic{
                context_.intrinsics().IsSpecificIntrinsicFunction(
                    interface->name().ToString())};
            !intrinsic || intrinsic->isRestrictedSpecific) { // C1515
          messages_.Say(
              "Intrinsic procedure '%s' is not an unrestricted specific "
              "intrinsic permitted for use as the definition of the interface "
              "to procedure pointer '%s'"_err_en_US,
              interface->name(), symbol.name());
        }
      } else if (interface->attrs().test(Attr::ELEMENTAL)) {
        messages_.Say("Procedure pointer '%s' may not be ELEMENTAL"_err_en_US,
            symbol.name()); // C1517
      }
    }
  } else if (symbol.attrs().test(Attr::SAVE)) {
    messages_.Say(
        "Procedure '%s' with SAVE attribute must also have POINTER attribute"_err_en_US,
        symbol.name());
  }
}

// When a module subprogram has the MODULE prefix the following must match
// with the corresponding separate module procedure interface body:
// - C1549: characteristics and dummy argument names
// - C1550: binding label
// - C1551: NON_RECURSIVE prefix
class SubprogramMatchHelper {
public:
  explicit SubprogramMatchHelper(CheckHelper &checkHelper)
      : checkHelper{checkHelper} {}

  void Check(const Symbol &, const Symbol &);

private:
  SemanticsContext &context() { return checkHelper.context(); }
  void CheckDummyArg(const Symbol &, const Symbol &, const DummyArgument &,
      const DummyArgument &);
  void CheckDummyDataObject(const Symbol &, const Symbol &,
      const DummyDataObject &, const DummyDataObject &);
  void CheckDummyProcedure(const Symbol &, const Symbol &,
      const DummyProcedure &, const DummyProcedure &);
  bool CheckSameIntent(
      const Symbol &, const Symbol &, common::Intent, common::Intent);
  template <typename... A>
  void Say(
      const Symbol &, const Symbol &, parser::MessageFixedText &&, A &&...);
  template <typename ATTRS>
  bool CheckSameAttrs(const Symbol &, const Symbol &, ATTRS, ATTRS);
  bool ShapesAreCompatible(const DummyDataObject &, const DummyDataObject &);
  evaluate::Shape FoldShape(const evaluate::Shape &);
  std::string AsFortran(DummyDataObject::Attr attr) {
    return parser::ToUpperCaseLetters(DummyDataObject::EnumToString(attr));
  }
  std::string AsFortran(DummyProcedure::Attr attr) {
    return parser::ToUpperCaseLetters(DummyProcedure::EnumToString(attr));
  }

  CheckHelper &checkHelper;
};

// 15.6.2.6 para 3 - can the result of an ENTRY differ from its function?
bool CheckHelper::IsResultOkToDiffer(const FunctionResult &result) {
  if (result.attrs.test(FunctionResult::Attr::Allocatable) ||
      result.attrs.test(FunctionResult::Attr::Pointer)) {
    return false;
  }
  const auto *typeAndShape{result.GetTypeAndShape()};
  if (!typeAndShape || typeAndShape->Rank() != 0) {
    return false;
  }
  auto category{typeAndShape->type().category()};
  if (category == TypeCategory::Character ||
      category == TypeCategory::Derived) {
    return false;
  }
  int kind{typeAndShape->type().kind()};
  return kind == context_.GetDefaultKind(category) ||
      (category == TypeCategory::Real &&
          kind == context_.doublePrecisionKind());
}

void CheckHelper::CheckSubprogram(
    const Symbol &symbol, const SubprogramDetails &details) {
  if (const Symbol * iface{FindSeparateModuleSubprogramInterface(&symbol)}) {
    SubprogramMatchHelper{*this}.Check(symbol, *iface);
  }
  if (const Scope * entryScope{details.entryScope()}) {
    // ENTRY 15.6.2.6, esp. C1571
    std::optional<parser::MessageFixedText> error;
    const Symbol *subprogram{entryScope->symbol()};
    const SubprogramDetails *subprogramDetails{nullptr};
    if (subprogram) {
      subprogramDetails = subprogram->detailsIf<SubprogramDetails>();
    }
    if (!(entryScope->parent().IsGlobal() || entryScope->parent().IsModule() ||
            entryScope->parent().IsSubmodule())) {
      error = "ENTRY may not appear in an internal subprogram"_err_en_US;
    } else if (subprogramDetails && details.isFunction() &&
        subprogramDetails->isFunction() &&
        !context_.HasError(details.result()) &&
        !context_.HasError(subprogramDetails->result())) {
      auto result{FunctionResult::Characterize(
          details.result(), context_.foldingContext())};
      auto subpResult{FunctionResult::Characterize(
          subprogramDetails->result(), context_.foldingContext())};
      if (result && subpResult && *result != *subpResult &&
          (!IsResultOkToDiffer(*result) || !IsResultOkToDiffer(*subpResult))) {
        error =
            "Result of ENTRY is not compatible with result of containing function"_err_en_US;
      }
    }
    if (error) {
      if (auto *msg{messages_.Say(symbol.name(), *error)}) {
        if (subprogram) {
          msg->Attach(subprogram->name(), "Containing subprogram"_en_US);
        }
      }
    }
  }
  if (symbol.attrs().test(Attr::ELEMENTAL)) {
    // See comment on the similar check in CheckProcEntity()
    if (details.isDummy()) {
      messages_.Say("A dummy procedure may not be ELEMENTAL"_err_en_US);
    } else {
      for (const Symbol *dummy : details.dummyArgs()) {
        if (!dummy) { // C15100
          messages_.Say(
              "An ELEMENTAL subroutine may not have an alternate return dummy argument"_err_en_US);
        }
      }
    }
  }
}

void CheckHelper::CheckDerivedType(
    const Symbol &derivedType, const DerivedTypeDetails &details) {
  if (details.isForwardReferenced() && !context_.HasError(derivedType)) {
    messages_.Say("The derived type '%s' has not been defined"_err_en_US,
        derivedType.name());
  }
  const Scope *scope{derivedType.scope()};
  if (!scope) {
    CHECK(details.isForwardReferenced());
    return;
  }
  CHECK(scope->symbol() == &derivedType);
  CHECK(scope->IsDerivedType());
  if (derivedType.attrs().test(Attr::ABSTRACT) && // C734
      (derivedType.attrs().test(Attr::BIND_C) || details.sequence())) {
    messages_.Say("An ABSTRACT derived type must be extensible"_err_en_US);
  }
  if (const DeclTypeSpec * parent{FindParentTypeSpec(derivedType)}) {
    const DerivedTypeSpec *parentDerived{parent->AsDerived()};
    if (!IsExtensibleType(parentDerived)) { // C705
      messages_.Say("The parent type is not extensible"_err_en_US);
    }
    if (!derivedType.attrs().test(Attr::ABSTRACT) && parentDerived &&
        parentDerived->typeSymbol().attrs().test(Attr::ABSTRACT)) {
      ScopeComponentIterator components{*parentDerived};
      for (const Symbol &component : components) {
        if (component.attrs().test(Attr::DEFERRED)) {
          if (scope->FindComponent(component.name()) == &component) {
            SayWithDeclaration(component,
                "Non-ABSTRACT extension of ABSTRACT derived type '%s' lacks a binding for DEFERRED procedure '%s'"_err_en_US,
                parentDerived->typeSymbol().name(), component.name());
          }
        }
      }
    }
    DerivedTypeSpec derived{derivedType.name(), derivedType};
    derived.set_scope(*scope);
    if (FindCoarrayUltimateComponent(derived) && // C736
        !(parentDerived && FindCoarrayUltimateComponent(*parentDerived))) {
      messages_.Say(
          "Type '%s' has a coarray ultimate component so the type at the base "
          "of its type extension chain ('%s') must be a type that has a "
          "coarray ultimate component"_err_en_US,
          derivedType.name(), scope->GetDerivedTypeBase().GetSymbol()->name());
    }
    if (FindEventOrLockPotentialComponent(derived) && // C737
        !(FindEventOrLockPotentialComponent(*parentDerived) ||
            IsEventTypeOrLockType(parentDerived))) {
      messages_.Say(
          "Type '%s' has an EVENT_TYPE or LOCK_TYPE component, so the type "
          "at the base of its type extension chain ('%s') must either have an "
          "EVENT_TYPE or LOCK_TYPE component, or be EVENT_TYPE or "
          "LOCK_TYPE"_err_en_US,
          derivedType.name(), scope->GetDerivedTypeBase().GetSymbol()->name());
    }
  }
  if (HasIntrinsicTypeName(derivedType)) { // C729
    messages_.Say("A derived type name cannot be the name of an intrinsic"
                  " type"_err_en_US);
  }
  std::map<SourceName, SymbolRef> previous;
  for (const auto &pair : details.finals()) {
    SourceName source{pair.first};
    const Symbol &ref{*pair.second};
    if (CheckFinal(ref, source, derivedType) &&
        std::all_of(previous.begin(), previous.end(),
            [&](std::pair<SourceName, SymbolRef> prev) {
              return CheckDistinguishableFinals(
                  ref, source, *prev.second, prev.first, derivedType);
            })) {
      previous.emplace(source, ref);
    }
  }
}

// C786
bool CheckHelper::CheckFinal(
    const Symbol &subroutine, SourceName finalName, const Symbol &derivedType) {
  if (!IsModuleProcedure(subroutine)) {
    SayWithDeclaration(subroutine, finalName,
        "FINAL subroutine '%s' of derived type '%s' must be a module procedure"_err_en_US,
        subroutine.name(), derivedType.name());
    return false;
  }
  const Procedure *proc{Characterize(subroutine)};
  if (!proc) {
    return false; // error recovery
  }
  if (!proc->IsSubroutine()) {
    SayWithDeclaration(subroutine, finalName,
        "FINAL subroutine '%s' of derived type '%s' must be a subroutine"_err_en_US,
        subroutine.name(), derivedType.name());
    return false;
  }
  if (proc->dummyArguments.size() != 1) {
    SayWithDeclaration(subroutine, finalName,
        "FINAL subroutine '%s' of derived type '%s' must have a single dummy argument"_err_en_US,
        subroutine.name(), derivedType.name());
    return false;
  }
  const auto &arg{proc->dummyArguments[0]};
  const Symbol *errSym{&subroutine};
  if (const auto *details{subroutine.detailsIf<SubprogramDetails>()}) {
    if (!details->dummyArgs().empty()) {
      if (const Symbol * argSym{details->dummyArgs()[0]}) {
        errSym = argSym;
      }
    }
  }
  const auto *ddo{std::get_if<DummyDataObject>(&arg.u)};
  if (!ddo) {
    SayWithDeclaration(subroutine, finalName,
        "FINAL subroutine '%s' of derived type '%s' must have a single dummy argument that is a data object"_err_en_US,
        subroutine.name(), derivedType.name());
    return false;
  }
  bool ok{true};
  if (arg.IsOptional()) {
    SayWithDeclaration(*errSym, finalName,
        "FINAL subroutine '%s' of derived type '%s' must not have an OPTIONAL dummy argument"_err_en_US,
        subroutine.name(), derivedType.name());
    ok = false;
  }
  if (ddo->attrs.test(DummyDataObject::Attr::Allocatable)) {
    SayWithDeclaration(*errSym, finalName,
        "FINAL subroutine '%s' of derived type '%s' must not have an ALLOCATABLE dummy argument"_err_en_US,
        subroutine.name(), derivedType.name());
    ok = false;
  }
  if (ddo->attrs.test(DummyDataObject::Attr::Pointer)) {
    SayWithDeclaration(*errSym, finalName,
        "FINAL subroutine '%s' of derived type '%s' must not have a POINTER dummy argument"_err_en_US,
        subroutine.name(), derivedType.name());
    ok = false;
  }
  if (ddo->intent == common::Intent::Out) {
    SayWithDeclaration(*errSym, finalName,
        "FINAL subroutine '%s' of derived type '%s' must not have a dummy argument with INTENT(OUT)"_err_en_US,
        subroutine.name(), derivedType.name());
    ok = false;
  }
  if (ddo->attrs.test(DummyDataObject::Attr::Value)) {
    SayWithDeclaration(*errSym, finalName,
        "FINAL subroutine '%s' of derived type '%s' must not have a dummy argument with the VALUE attribute"_err_en_US,
        subroutine.name(), derivedType.name());
    ok = false;
  }
  if (ddo->type.corank() > 0) {
    SayWithDeclaration(*errSym, finalName,
        "FINAL subroutine '%s' of derived type '%s' must not have a coarray dummy argument"_err_en_US,
        subroutine.name(), derivedType.name());
    ok = false;
  }
  if (ddo->type.type().IsPolymorphic()) {
    SayWithDeclaration(*errSym, finalName,
        "FINAL subroutine '%s' of derived type '%s' must not have a polymorphic dummy argument"_err_en_US,
        subroutine.name(), derivedType.name());
    ok = false;
  } else if (ddo->type.type().category() != TypeCategory::Derived ||
      &ddo->type.type().GetDerivedTypeSpec().typeSymbol() != &derivedType) {
    SayWithDeclaration(*errSym, finalName,
        "FINAL subroutine '%s' of derived type '%s' must have a TYPE(%s) dummy argument"_err_en_US,
        subroutine.name(), derivedType.name(), derivedType.name());
    ok = false;
  } else { // check that all LEN type parameters are assumed
    for (auto ref : OrderParameterDeclarations(derivedType)) {
      if (IsLenTypeParameter(*ref)) {
        const auto *value{
            ddo->type.type().GetDerivedTypeSpec().FindParameter(ref->name())};
        if (!value || !value->isAssumed()) {
          SayWithDeclaration(*errSym, finalName,
              "FINAL subroutine '%s' of derived type '%s' must have a dummy argument with an assumed LEN type parameter '%s=*'"_err_en_US,
              subroutine.name(), derivedType.name(), ref->name());
          ok = false;
        }
      }
    }
  }
  return ok;
}

bool CheckHelper::CheckDistinguishableFinals(const Symbol &f1,
    SourceName f1Name, const Symbol &f2, SourceName f2Name,
    const Symbol &derivedType) {
  const Procedure *p1{Characterize(f1)};
  const Procedure *p2{Characterize(f2)};
  if (p1 && p2) {
    if (characteristics::Distinguishable(
            context_.languageFeatures(), *p1, *p2)) {
      return true;
    }
    if (auto *msg{messages_.Say(f1Name,
            "FINAL subroutines '%s' and '%s' of derived type '%s' cannot be distinguished by rank or KIND type parameter value"_err_en_US,
            f1Name, f2Name, derivedType.name())}) {
      msg->Attach(f2Name, "FINAL declaration of '%s'"_en_US, f2.name())
          .Attach(f1.name(), "Definition of '%s'"_en_US, f1Name)
          .Attach(f2.name(), "Definition of '%s'"_en_US, f2Name);
    }
  }
  return false;
}

void CheckHelper::CheckHostAssoc(
    const Symbol &symbol, const HostAssocDetails &details) {
  const Symbol &hostSymbol{details.symbol()};
  if (hostSymbol.test(Symbol::Flag::ImplicitOrError)) {
    if (details.implicitOrSpecExprError) {
      messages_.Say("Implicitly typed local entity '%s' not allowed in"
                    " specification expression"_err_en_US,
          symbol.name());
    } else if (details.implicitOrExplicitTypeError) {
      messages_.Say(
          "No explicit type declared for '%s'"_err_en_US, symbol.name());
    }
  }
}

void CheckHelper::CheckGeneric(
    const Symbol &symbol, const GenericDetails &details) {
  CheckSpecificsAreDistinguishable(symbol, details);
  common::visit(common::visitors{
                    [&](const GenericKind::DefinedIo &io) {
                      CheckDefinedIoProc(symbol, details, io);
                    },
                    [&](const GenericKind::OtherKind &other) {
                      if (other == GenericKind::OtherKind::Name) {
                        CheckGenericVsIntrinsic(symbol, details);
                      }
                    },
                    [](const auto &) {},
                },
      details.kind().u);
}

// Check that the specifics of this generic are distinguishable from each other
void CheckHelper::CheckSpecificsAreDistinguishable(
    const Symbol &generic, const GenericDetails &details) {
  GenericKind kind{details.kind()};
  const SymbolVector &specifics{details.specificProcs()};
  std::size_t count{specifics.size()};
  if (count < 2 || !kind.IsName()) {
    return;
  }
  DistinguishabilityHelper helper{context_};
  for (const Symbol &specific : specifics) {
    if (const Procedure * procedure{Characterize(specific)}) {
      helper.Add(generic, kind, specific, *procedure);
    }
  }
  helper.Check(generic.owner());
}

static bool ConflictsWithIntrinsicAssignment(const Procedure &proc) {
  auto lhs{std::get<DummyDataObject>(proc.dummyArguments[0].u).type};
  auto rhs{std::get<DummyDataObject>(proc.dummyArguments[1].u).type};
  return Tristate::No ==
      IsDefinedAssignment(lhs.type(), lhs.Rank(), rhs.type(), rhs.Rank());
}

static bool ConflictsWithIntrinsicOperator(
    const GenericKind &kind, const Procedure &proc) {
  if (!kind.IsIntrinsicOperator()) {
    return false;
  }
  auto arg0{std::get<DummyDataObject>(proc.dummyArguments[0].u).type};
  auto type0{arg0.type()};
  if (proc.dummyArguments.size() == 1) { // unary
    return common::visit(
        common::visitors{
            [&](common::NumericOperator) { return IsIntrinsicNumeric(type0); },
            [&](common::LogicalOperator) { return IsIntrinsicLogical(type0); },
            [](const auto &) -> bool { DIE("bad generic kind"); },
        },
        kind.u);
  } else { // binary
    int rank0{arg0.Rank()};
    auto arg1{std::get<DummyDataObject>(proc.dummyArguments[1].u).type};
    auto type1{arg1.type()};
    int rank1{arg1.Rank()};
    return common::visit(
        common::visitors{
            [&](common::NumericOperator) {
              return IsIntrinsicNumeric(type0, rank0, type1, rank1);
            },
            [&](common::LogicalOperator) {
              return IsIntrinsicLogical(type0, rank0, type1, rank1);
            },
            [&](common::RelationalOperator opr) {
              return IsIntrinsicRelational(opr, type0, rank0, type1, rank1);
            },
            [&](GenericKind::OtherKind x) {
              CHECK(x == GenericKind::OtherKind::Concat);
              return IsIntrinsicConcat(type0, rank0, type1, rank1);
            },
            [](const auto &) -> bool { DIE("bad generic kind"); },
        },
        kind.u);
  }
}

// Check if this procedure can be used for defined operators (see 15.4.3.4.2).
bool CheckHelper::CheckDefinedOperator(SourceName opName, GenericKind kind,
    const Symbol &specific, const Procedure &proc) {
  if (context_.HasError(specific)) {
    return false;
  }
  std::optional<parser::MessageFixedText> msg;
  auto checkDefinedOperatorArgs{
      [&](SourceName opName, const Symbol &specific, const Procedure &proc) {
        bool arg0Defined{CheckDefinedOperatorArg(opName, specific, proc, 0)};
        bool arg1Defined{CheckDefinedOperatorArg(opName, specific, proc, 1)};
        return arg0Defined && arg1Defined;
      }};
  if (specific.attrs().test(Attr::NOPASS)) { // C774
    msg = "%s procedure '%s' may not have NOPASS attribute"_err_en_US;
  } else if (!proc.functionResult.has_value()) {
    msg = "%s procedure '%s' must be a function"_err_en_US;
  } else if (proc.functionResult->IsAssumedLengthCharacter()) {
    msg = "%s function '%s' may not have assumed-length CHARACTER(*)"
          " result"_err_en_US;
  } else if (auto m{CheckNumberOfArgs(kind, proc.dummyArguments.size())}) {
    msg = std::move(m);
  } else if (!checkDefinedOperatorArgs(opName, specific, proc)) {
    return false; // error was reported
  } else if (ConflictsWithIntrinsicOperator(kind, proc)) {
    msg = "%s function '%s' conflicts with intrinsic operator"_err_en_US;
  } else {
    return true; // OK
  }
  SayWithDeclaration(
      specific, std::move(*msg), MakeOpName(opName), specific.name());
  context_.SetError(specific);
  return false;
}

// If the number of arguments is wrong for this intrinsic operator, return
// false and return the error message in msg.
std::optional<parser::MessageFixedText> CheckHelper::CheckNumberOfArgs(
    const GenericKind &kind, std::size_t nargs) {
  if (!kind.IsIntrinsicOperator()) {
    return std::nullopt;
  }
  std::size_t min{2}, max{2}; // allowed number of args; default is binary
  common::visit(common::visitors{
                    [&](const common::NumericOperator &x) {
                      if (x == common::NumericOperator::Add ||
                          x == common::NumericOperator::Subtract) {
                        min = 1; // + and - are unary or binary
                      }
                    },
                    [&](const common::LogicalOperator &x) {
                      if (x == common::LogicalOperator::Not) {
                        min = 1; // .NOT. is unary
                        max = 1;
                      }
                    },
                    [](const common::RelationalOperator &) {
                      // all are binary
                    },
                    [](const GenericKind::OtherKind &x) {
                      CHECK(x == GenericKind::OtherKind::Concat);
                    },
                    [](const auto &) { DIE("expected intrinsic operator"); },
                },
      kind.u);
  if (nargs >= min && nargs <= max) {
    return std::nullopt;
  } else if (max == 1) {
    return "%s function '%s' must have one dummy argument"_err_en_US;
  } else if (min == 2) {
    return "%s function '%s' must have two dummy arguments"_err_en_US;
  } else {
    return "%s function '%s' must have one or two dummy arguments"_err_en_US;
  }
}

bool CheckHelper::CheckDefinedOperatorArg(const SourceName &opName,
    const Symbol &symbol, const Procedure &proc, std::size_t pos) {
  if (pos >= proc.dummyArguments.size()) {
    return true;
  }
  auto &arg{proc.dummyArguments.at(pos)};
  std::optional<parser::MessageFixedText> msg;
  if (arg.IsOptional()) {
    msg = "In %s function '%s', dummy argument '%s' may not be"
          " OPTIONAL"_err_en_US;
  } else if (const auto *dataObject{std::get_if<DummyDataObject>(&arg.u)};
             dataObject == nullptr) {
    msg = "In %s function '%s', dummy argument '%s' must be a"
          " data object"_err_en_US;
  } else if (dataObject->intent != common::Intent::In &&
      !dataObject->attrs.test(DummyDataObject::Attr::Value)) {
    msg = "In %s function '%s', dummy argument '%s' must have INTENT(IN)"
          " or VALUE attribute"_err_en_US;
  }
  if (msg) {
    SayWithDeclaration(symbol, std::move(*msg),
        parser::ToUpperCaseLetters(opName.ToString()), symbol.name(), arg.name);
    return false;
  }
  return true;
}

// Check if this procedure can be used for defined assignment (see 15.4.3.4.3).
bool CheckHelper::CheckDefinedAssignment(
    const Symbol &specific, const Procedure &proc) {
  if (context_.HasError(specific)) {
    return false;
  }
  std::optional<parser::MessageFixedText> msg;
  if (specific.attrs().test(Attr::NOPASS)) { // C774
    msg = "Defined assignment procedure '%s' may not have"
          " NOPASS attribute"_err_en_US;
  } else if (!proc.IsSubroutine()) {
    msg = "Defined assignment procedure '%s' must be a subroutine"_err_en_US;
  } else if (proc.dummyArguments.size() != 2) {
    msg = "Defined assignment subroutine '%s' must have"
          " two dummy arguments"_err_en_US;
  } else {
    // Check both arguments even if the first has an error.
    bool ok0{CheckDefinedAssignmentArg(specific, proc.dummyArguments[0], 0)};
    bool ok1{CheckDefinedAssignmentArg(specific, proc.dummyArguments[1], 1)};
    if (!(ok0 && ok1)) {
      return false; // error was reported
    } else if (ConflictsWithIntrinsicAssignment(proc)) {
      msg = "Defined assignment subroutine '%s' conflicts with"
            " intrinsic assignment"_err_en_US;
    } else {
      return true; // OK
    }
  }
  SayWithDeclaration(specific, std::move(msg.value()), specific.name());
  context_.SetError(specific);
  return false;
}

bool CheckHelper::CheckDefinedAssignmentArg(
    const Symbol &symbol, const DummyArgument &arg, int pos) {
  std::optional<parser::MessageFixedText> msg;
  if (arg.IsOptional()) {
    msg = "In defined assignment subroutine '%s', dummy argument '%s'"
          " may not be OPTIONAL"_err_en_US;
  } else if (const auto *dataObject{std::get_if<DummyDataObject>(&arg.u)}) {
    if (pos == 0) {
      if (dataObject->intent != common::Intent::Out &&
          dataObject->intent != common::Intent::InOut) {
        msg = "In defined assignment subroutine '%s', first dummy argument '%s'"
              " must have INTENT(OUT) or INTENT(INOUT)"_err_en_US;
      }
    } else if (pos == 1) {
      if (dataObject->intent != common::Intent::In &&
          !dataObject->attrs.test(DummyDataObject::Attr::Value)) {
        msg =
            "In defined assignment subroutine '%s', second dummy"
            " argument '%s' must have INTENT(IN) or VALUE attribute"_err_en_US;
      }
    } else {
      DIE("pos must be 0 or 1");
    }
  } else {
    msg = "In defined assignment subroutine '%s', dummy argument '%s'"
          " must be a data object"_err_en_US;
  }
  if (msg) {
    SayWithDeclaration(symbol, std::move(*msg), symbol.name(), arg.name);
    context_.SetError(symbol);
    return false;
  }
  return true;
}

// Report a conflicting attribute error if symbol has both of these attributes
bool CheckHelper::CheckConflicting(const Symbol &symbol, Attr a1, Attr a2) {
  if (symbol.attrs().test(a1) && symbol.attrs().test(a2)) {
    messages_.Say("'%s' may not have both the %s and %s attributes"_err_en_US,
        symbol.name(), AttrToString(a1), AttrToString(a2));
    return true;
  } else {
    return false;
  }
}

void CheckHelper::WarnMissingFinal(const Symbol &symbol) {
  const auto *object{symbol.detailsIf<ObjectEntityDetails>()};
  if (!object || IsPointer(symbol)) {
    return;
  }
  const DeclTypeSpec *type{object->type()};
  const DerivedTypeSpec *derived{type ? type->AsDerived() : nullptr};
  const Symbol *derivedSym{derived ? &derived->typeSymbol() : nullptr};
  int rank{object->shape().Rank()};
  const Symbol *initialDerivedSym{derivedSym};
  while (const auto *derivedDetails{
      derivedSym ? derivedSym->detailsIf<DerivedTypeDetails>() : nullptr}) {
    if (!derivedDetails->finals().empty() &&
        !derivedDetails->GetFinalForRank(rank)) {
      if (auto *msg{derivedSym == initialDerivedSym
                  ? messages_.Say(symbol.name(),
                        "'%s' of derived type '%s' does not have a FINAL subroutine for its rank (%d)"_warn_en_US,
                        symbol.name(), derivedSym->name(), rank)
                  : messages_.Say(symbol.name(),
                        "'%s' of derived type '%s' extended from '%s' does not have a FINAL subroutine for its rank (%d)"_warn_en_US,
                        symbol.name(), initialDerivedSym->name(),
                        derivedSym->name(), rank)}) {
        msg->Attach(derivedSym->name(),
            "Declaration of derived type '%s'"_en_US, derivedSym->name());
      }
      return;
    }
    derived = derivedSym->GetParentTypeSpec();
    derivedSym = derived ? &derived->typeSymbol() : nullptr;
  }
}

const Procedure *CheckHelper::Characterize(const Symbol &symbol) {
  auto it{characterizeCache_.find(symbol)};
  if (it == characterizeCache_.end()) {
    auto pair{characterizeCache_.emplace(SymbolRef{symbol},
        Procedure::Characterize(symbol, context_.foldingContext()))};
    it = pair.first;
  }
  return common::GetPtrFromOptional(it->second);
}

void CheckHelper::CheckVolatile(const Symbol &symbol,
    const DerivedTypeSpec *derived) { // C866 - C868
  if (IsIntentIn(symbol)) {
    messages_.Say(
        "VOLATILE attribute may not apply to an INTENT(IN) argument"_err_en_US);
  }
  if (IsProcedure(symbol)) {
    messages_.Say("VOLATILE attribute may apply only to a variable"_err_en_US);
  }
  if (symbol.has<UseDetails>() || symbol.has<HostAssocDetails>()) {
    const Symbol &ultimate{symbol.GetUltimate()};
    if (evaluate::IsCoarray(ultimate)) {
      messages_.Say(
          "VOLATILE attribute may not apply to a coarray accessed by USE or host association"_err_en_US);
    }
    if (derived) {
      if (FindCoarrayUltimateComponent(*derived)) {
        messages_.Say(
            "VOLATILE attribute may not apply to a type with a coarray ultimate component accessed by USE or host association"_err_en_US);
      }
    }
  }
}

void CheckHelper::CheckPointer(const Symbol &symbol) { // C852
  CheckConflicting(symbol, Attr::POINTER, Attr::TARGET);
  CheckConflicting(symbol, Attr::POINTER, Attr::ALLOCATABLE); // C751
  CheckConflicting(symbol, Attr::POINTER, Attr::INTRINSIC);
  // Prohibit constant pointers.  The standard does not explicitly prohibit
  // them, but the PARAMETER attribute requires a entity-decl to have an
  // initialization that is a constant-expr, and the only form of
  // initialization that allows a constant-expr is the one that's not a "=>"
  // pointer initialization.  See C811, C807, and section 8.5.13.
  CheckConflicting(symbol, Attr::POINTER, Attr::PARAMETER);
  if (symbol.Corank() > 0) {
    messages_.Say(
        "'%s' may not have the POINTER attribute because it is a coarray"_err_en_US,
        symbol.name());
  }
}

// C760 constraints on the passed-object dummy argument
// C757 constraints on procedure pointer components
void CheckHelper::CheckPassArg(
    const Symbol &proc, const Symbol *interface, const WithPassArg &details) {
  if (proc.attrs().test(Attr::NOPASS)) {
    return;
  }
  const auto &name{proc.name()};
  if (!interface) {
    messages_.Say(name,
        "Procedure component '%s' must have NOPASS attribute or explicit interface"_err_en_US,
        name);
    return;
  }
  const auto *subprogram{interface->detailsIf<SubprogramDetails>()};
  if (!subprogram) {
    messages_.Say(name,
        "Procedure component '%s' has invalid interface '%s'"_err_en_US, name,
        interface->name());
    return;
  }
  std::optional<SourceName> passName{details.passName()};
  const auto &dummyArgs{subprogram->dummyArgs()};
  if (!passName) {
    if (dummyArgs.empty()) {
      messages_.Say(name,
          proc.has<ProcEntityDetails>()
              ? "Procedure component '%s' with no dummy arguments"
                " must have NOPASS attribute"_err_en_US
              : "Procedure binding '%s' with no dummy arguments"
                " must have NOPASS attribute"_err_en_US,
          name);
      context_.SetError(*interface);
      return;
    }
    Symbol *argSym{dummyArgs[0]};
    if (!argSym) {
      messages_.Say(interface->name(),
          "Cannot use an alternate return as the passed-object dummy "
          "argument"_err_en_US);
      return;
    }
    passName = dummyArgs[0]->name();
  }
  std::optional<int> passArgIndex{};
  for (std::size_t i{0}; i < dummyArgs.size(); ++i) {
    if (dummyArgs[i] && dummyArgs[i]->name() == *passName) {
      passArgIndex = i;
      break;
    }
  }
  if (!passArgIndex) { // C758
    messages_.Say(*passName,
        "'%s' is not a dummy argument of procedure interface '%s'"_err_en_US,
        *passName, interface->name());
    return;
  }
  const Symbol &passArg{*dummyArgs[*passArgIndex]};
  std::optional<parser::MessageFixedText> msg;
  if (!passArg.has<ObjectEntityDetails>()) {
    msg = "Passed-object dummy argument '%s' of procedure '%s'"
          " must be a data object"_err_en_US;
  } else if (passArg.attrs().test(Attr::POINTER)) {
    msg = "Passed-object dummy argument '%s' of procedure '%s'"
          " may not have the POINTER attribute"_err_en_US;
  } else if (passArg.attrs().test(Attr::ALLOCATABLE)) {
    msg = "Passed-object dummy argument '%s' of procedure '%s'"
          " may not have the ALLOCATABLE attribute"_err_en_US;
  } else if (passArg.attrs().test(Attr::VALUE)) {
    msg = "Passed-object dummy argument '%s' of procedure '%s'"
          " may not have the VALUE attribute"_err_en_US;
  } else if (passArg.Rank() > 0) {
    msg = "Passed-object dummy argument '%s' of procedure '%s'"
          " must be scalar"_err_en_US;
  }
  if (msg) {
    messages_.Say(name, std::move(*msg), passName.value(), name);
    return;
  }
  const DeclTypeSpec *type{passArg.GetType()};
  if (!type) {
    return; // an error already occurred
  }
  const Symbol &typeSymbol{*proc.owner().GetSymbol()};
  const DerivedTypeSpec *derived{type->AsDerived()};
  if (!derived || derived->typeSymbol() != typeSymbol) {
    messages_.Say(name,
        "Passed-object dummy argument '%s' of procedure '%s'"
        " must be of type '%s' but is '%s'"_err_en_US,
        passName.value(), name, typeSymbol.name(), type->AsFortran());
    return;
  }
  if (IsExtensibleType(derived) != type->IsPolymorphic()) {
    messages_.Say(name,
        type->IsPolymorphic()
            ? "Passed-object dummy argument '%s' of procedure '%s'"
              " may not be polymorphic because '%s' is not extensible"_err_en_US
            : "Passed-object dummy argument '%s' of procedure '%s'"
              " must be polymorphic because '%s' is extensible"_err_en_US,
        passName.value(), name, typeSymbol.name());
    return;
  }
  for (const auto &[paramName, paramValue] : derived->parameters()) {
    if (paramValue.isLen() && !paramValue.isAssumed()) {
      messages_.Say(name,
          "Passed-object dummy argument '%s' of procedure '%s'"
          " has non-assumed length parameter '%s'"_err_en_US,
          passName.value(), name, paramName);
    }
  }
}

void CheckHelper::CheckProcBinding(
    const Symbol &symbol, const ProcBindingDetails &binding) {
  const Scope &dtScope{symbol.owner()};
  CHECK(dtScope.kind() == Scope::Kind::DerivedType);
  if (symbol.attrs().test(Attr::DEFERRED)) {
    if (const Symbol * dtSymbol{dtScope.symbol()}) {
      if (!dtSymbol->attrs().test(Attr::ABSTRACT)) { // C733
        SayWithDeclaration(*dtSymbol,
            "Procedure bound to non-ABSTRACT derived type '%s' may not be DEFERRED"_err_en_US,
            dtSymbol->name());
      }
    }
    if (symbol.attrs().test(Attr::NON_OVERRIDABLE)) {
      messages_.Say(
          "Type-bound procedure '%s' may not be both DEFERRED and NON_OVERRIDABLE"_err_en_US,
          symbol.name());
    }
  }
  if (binding.symbol().attrs().test(Attr::INTRINSIC) &&
      !context_.intrinsics().IsSpecificIntrinsicFunction(
          binding.symbol().name().ToString())) {
    messages_.Say(
        "Intrinsic procedure '%s' is not a specific intrinsic permitted for use in the definition of binding '%s'"_err_en_US,
        binding.symbol().name(), symbol.name());
  }
  if (const Symbol * overridden{FindOverriddenBinding(symbol)}) {
    if (overridden->attrs().test(Attr::NON_OVERRIDABLE)) {
      SayWithDeclaration(*overridden,
          "Override of NON_OVERRIDABLE '%s' is not permitted"_err_en_US,
          symbol.name());
    }
    if (const auto *overriddenBinding{
            overridden->detailsIf<ProcBindingDetails>()}) {
      if (!IsPureProcedure(symbol) && IsPureProcedure(*overridden)) {
        SayWithDeclaration(*overridden,
            "An overridden pure type-bound procedure binding must also be pure"_err_en_US);
        return;
      }
      if (!binding.symbol().attrs().test(Attr::ELEMENTAL) &&
          overriddenBinding->symbol().attrs().test(Attr::ELEMENTAL)) {
        SayWithDeclaration(*overridden,
            "A type-bound procedure and its override must both, or neither, be ELEMENTAL"_err_en_US);
        return;
      }
      bool isNopass{symbol.attrs().test(Attr::NOPASS)};
      if (isNopass != overridden->attrs().test(Attr::NOPASS)) {
        SayWithDeclaration(*overridden,
            isNopass
                ? "A NOPASS type-bound procedure may not override a passed-argument procedure"_err_en_US
                : "A passed-argument type-bound procedure may not override a NOPASS procedure"_err_en_US);
      } else {
        const auto *bindingChars{Characterize(binding.symbol())};
        const auto *overriddenChars{Characterize(overriddenBinding->symbol())};
        if (bindingChars && overriddenChars) {
          if (isNopass) {
            if (!bindingChars->CanOverride(*overriddenChars, std::nullopt)) {
              SayWithDeclaration(*overridden,
                  "A type-bound procedure and its override must have compatible interfaces"_err_en_US);
            }
          } else if (!context_.HasError(binding.symbol())) {
            int passIndex{bindingChars->FindPassIndex(binding.passName())};
            int overriddenPassIndex{
                overriddenChars->FindPassIndex(overriddenBinding->passName())};
            if (passIndex != overriddenPassIndex) {
              SayWithDeclaration(*overridden,
                  "A type-bound procedure and its override must use the same PASS argument"_err_en_US);
            } else if (!bindingChars->CanOverride(
                           *overriddenChars, passIndex)) {
              SayWithDeclaration(*overridden,
                  "A type-bound procedure and its override must have compatible interfaces apart from their passed argument"_err_en_US);
            }
          }
        }
      }
      if (symbol.attrs().test(Attr::PRIVATE) &&
          overridden->attrs().test(Attr::PUBLIC)) {
        SayWithDeclaration(*overridden,
            "A PRIVATE procedure may not override a PUBLIC procedure"_err_en_US);
      }
    } else {
      SayWithDeclaration(*overridden,
          "A type-bound procedure binding may not have the same name as a parent component"_err_en_US);
    }
  }
  CheckPassArg(symbol, &binding.symbol(), binding);
}

void CheckHelper::Check(const Scope &scope) {
  scope_ = &scope;
  common::Restorer<const Symbol *> restorer{innermostSymbol_, innermostSymbol_};
  if (const Symbol * symbol{scope.symbol()}) {
    innermostSymbol_ = symbol;
  }
  if (scope.IsParameterizedDerivedTypeInstantiation()) {
    auto restorer{common::ScopedSet(scopeIsUninstantiatedPDT_, false)};
    auto restorer2{context_.foldingContext().messages().SetContext(
        scope.instantiationContext().get())};
    for (const auto &pair : scope) {
      CheckPointerInitialization(*pair.second);
    }
  } else {
    auto restorer{common::ScopedSet(
        scopeIsUninstantiatedPDT_, scope.IsParameterizedDerivedType())};
    for (const auto &set : scope.equivalenceSets()) {
      CheckEquivalenceSet(set);
    }
    for (const auto &pair : scope) {
      Check(*pair.second);
    }
    int mainProgCnt{0};
    for (const Scope &child : scope.children()) {
      Check(child);
      // A program shall consist of exactly one main program (5.2.2).
      if (child.kind() == Scope::Kind::MainProgram) {
        ++mainProgCnt;
        if (mainProgCnt > 1) {
          messages_.Say(child.sourceRange(),
              "A source file cannot contain more than one main program"_err_en_US);
        }
      }
    }
    if (scope.kind() == Scope::Kind::BlockData) {
      CheckBlockData(scope);
    }
    CheckGenericOps(scope);
  }
}

void CheckHelper::CheckEquivalenceSet(const EquivalenceSet &set) {
  auto iter{
      std::find_if(set.begin(), set.end(), [](const EquivalenceObject &object) {
        return FindCommonBlockContaining(object.symbol) != nullptr;
      })};
  if (iter != set.end()) {
    const Symbol &commonBlock{DEREF(FindCommonBlockContaining(iter->symbol))};
    for (auto &object : set) {
      if (&object != &*iter) {
        if (auto *details{object.symbol.detailsIf<ObjectEntityDetails>()}) {
          if (details->commonBlock()) {
            if (details->commonBlock() != &commonBlock) { // 8.10.3 paragraph 1
              if (auto *msg{messages_.Say(object.symbol.name(),
                      "Two objects in the same EQUIVALENCE set may not be members of distinct COMMON blocks"_err_en_US)}) {
                msg->Attach(iter->symbol.name(),
                       "Other object in EQUIVALENCE set"_en_US)
                    .Attach(details->commonBlock()->name(),
                        "COMMON block containing '%s'"_en_US,
                        object.symbol.name())
                    .Attach(commonBlock.name(),
                        "COMMON block containing '%s'"_en_US,
                        iter->symbol.name());
              }
            }
          } else {
            // Mark all symbols in the equivalence set with the same COMMON
            // block to prevent spurious error messages about initialization
            // in BLOCK DATA outside COMMON
            details->set_commonBlock(commonBlock);
          }
        }
      }
    }
  }
  // TODO: Move C8106 (&al.) checks here from resolve-names-utils.cpp
}

void CheckHelper::CheckBlockData(const Scope &scope) {
  // BLOCK DATA subprograms should contain only named common blocks.
  // C1415 presents a list of statements that shouldn't appear in
  // BLOCK DATA, but so long as the subprogram contains no executable
  // code and allocates no storage outside named COMMON, we're happy
  // (e.g., an ENUM is strictly not allowed).
  for (const auto &pair : scope) {
    const Symbol &symbol{*pair.second};
    if (!(symbol.has<CommonBlockDetails>() || symbol.has<UseDetails>() ||
            symbol.has<UseErrorDetails>() || symbol.has<DerivedTypeDetails>() ||
            symbol.has<SubprogramDetails>() ||
            symbol.has<ObjectEntityDetails>() ||
            (symbol.has<ProcEntityDetails>() &&
                !symbol.attrs().test(Attr::POINTER)))) {
      messages_.Say(symbol.name(),
          "'%s' may not appear in a BLOCK DATA subprogram"_err_en_US,
          symbol.name());
    }
  }
}

// Check distinguishability of generic assignment and operators.
// For these, generics and generic bindings must be considered together.
void CheckHelper::CheckGenericOps(const Scope &scope) {
  DistinguishabilityHelper helper{context_};
  auto addSpecifics{[&](const Symbol &generic) {
    const auto *details{generic.GetUltimate().detailsIf<GenericDetails>()};
    if (!details) {
      // Not a generic; ensure characteristics are defined if a function.
      auto restorer{messages_.SetLocation(generic.name())};
      if (IsFunction(generic) && !context_.HasError(generic)) {
        if (const Symbol * result{FindFunctionResult(generic)};
            result && !context_.HasError(*result)) {
          Characterize(generic);
        }
      }
      return;
    }
    GenericKind kind{details->kind()};
    if (!kind.IsAssignment() && !kind.IsOperator()) {
      return;
    }
    const SymbolVector &specifics{details->specificProcs()};
    const std::vector<SourceName> &bindingNames{details->bindingNames()};
    for (std::size_t i{0}; i < specifics.size(); ++i) {
      const Symbol &specific{*specifics[i]};
      auto restorer{messages_.SetLocation(bindingNames[i])};
      if (const Procedure * proc{Characterize(specific)}) {
        if (kind.IsAssignment()) {
          if (!CheckDefinedAssignment(specific, *proc)) {
            continue;
          }
        } else {
          if (!CheckDefinedOperator(generic.name(), kind, specific, *proc)) {
            continue;
          }
        }
        helper.Add(generic, kind, specific, *proc);
      }
    }
  }};
  for (const auto &pair : scope) {
    const Symbol &symbol{*pair.second};
    addSpecifics(symbol);
    const Symbol &ultimate{symbol.GetUltimate()};
    if (ultimate.has<DerivedTypeDetails>()) {
      if (const Scope * typeScope{ultimate.scope()}) {
        for (const auto &pair2 : *typeScope) {
          addSpecifics(*pair2.second);
        }
      }
    }
  }
  helper.Check(scope);
}

static const std::string *DefinesBindCName(const Symbol &symbol) {
  const auto *subp{symbol.detailsIf<SubprogramDetails>()};
  if ((subp && !subp->isInterface() &&
          ClassifyProcedure(symbol) != ProcedureDefinitionClass::Internal) ||
      symbol.has<ObjectEntityDetails>()) {
    // Symbol defines data or entry point
    return symbol.GetBindName();
  } else {
    return nullptr;
  }
}

// Check that BIND(C) names are distinct
void CheckHelper::CheckBindCName(const Symbol &symbol) {
  if (const std::string * name{DefinesBindCName(symbol)}) {
    auto pair{bindC_.emplace(*name, symbol)};
    if (!pair.second) {
      const Symbol &other{*pair.first->second};
      if (DefinesBindCName(other) && !context_.HasError(other)) {
        if (auto *msg{messages_.Say(
                "Two symbols have the same BIND(C) name '%s'"_err_en_US,
                *name)}) {
          msg->Attach(other.name(), "Conflicting symbol"_en_US);
        }
        context_.SetError(symbol);
        context_.SetError(other);
      }
    }
  }
}

bool CheckHelper::CheckDioDummyIsData(
    const Symbol &subp, const Symbol *arg, std::size_t position) {
  if (arg && arg->detailsIf<ObjectEntityDetails>()) {
    return true;
  } else {
    if (arg) {
      messages_.Say(arg->name(),
          "Dummy argument '%s' must be a data object"_err_en_US, arg->name());
    } else {
      messages_.Say(subp.name(),
          "Dummy argument %d of '%s' must be a data object"_err_en_US, position,
          subp.name());
    }
    return false;
  }
}

void CheckHelper::CheckAlreadySeenDefinedIo(const DerivedTypeSpec *derivedType,
    GenericKind::DefinedIo ioKind, const Symbol &proc) {
  for (TypeWithDefinedIo definedIoType : seenDefinedIoTypes_) {
    if (*derivedType == *definedIoType.type && ioKind == definedIoType.ioKind &&
        proc != definedIoType.proc) {
      SayWithDeclaration(proc, definedIoType.proc.name(),
          "Derived type '%s' already has defined input/output procedure"
          " '%s'"_err_en_US,
          derivedType->name(),
          parser::ToUpperCaseLetters(GenericKind::EnumToString(ioKind)));
      return;
    }
  }
  seenDefinedIoTypes_.emplace_back(
      TypeWithDefinedIo{derivedType, ioKind, proc});
}

void CheckHelper::CheckDioDummyIsDerived(
    const Symbol &subp, const Symbol &arg, GenericKind::DefinedIo ioKind) {
  if (const DeclTypeSpec * type{arg.GetType()}) {
    if (const DerivedTypeSpec * derivedType{type->AsDerived()}) {
      CheckAlreadySeenDefinedIo(derivedType, ioKind, subp);
      bool isPolymorphic{type->IsPolymorphic()};
      if (isPolymorphic != IsExtensibleType(derivedType)) {
        messages_.Say(arg.name(),
            "Dummy argument '%s' of a defined input/output procedure must be %s when the derived type is %s"_err_en_US,
            arg.name(), isPolymorphic ? "TYPE()" : "CLASS()",
            isPolymorphic ? "not extensible" : "extensible");
      }
    } else {
      messages_.Say(arg.name(),
          "Dummy argument '%s' of a defined input/output procedure must have a"
          " derived type"_err_en_US,
          arg.name());
    }
  }
}

void CheckHelper::CheckDioDummyIsDefaultInteger(
    const Symbol &subp, const Symbol &arg) {
  if (const DeclTypeSpec * type{arg.GetType()};
      type && type->IsNumeric(TypeCategory::Integer)) {
    if (const auto kind{evaluate::ToInt64(type->numericTypeSpec().kind())};
        kind && *kind == context_.GetDefaultKind(TypeCategory::Integer)) {
      return;
    }
  }
  messages_.Say(arg.name(),
      "Dummy argument '%s' of a defined input/output procedure"
      " must be an INTEGER of default KIND"_err_en_US,
      arg.name());
}

void CheckHelper::CheckDioDummyIsScalar(const Symbol &subp, const Symbol &arg) {
  if (arg.Rank() > 0 || arg.Corank() > 0) {
    messages_.Say(arg.name(),
        "Dummy argument '%s' of a defined input/output procedure"
        " must be a scalar"_err_en_US,
        arg.name());
  }
}

void CheckHelper::CheckDioDtvArg(
    const Symbol &subp, const Symbol *arg, GenericKind::DefinedIo ioKind) {
  // Dtv argument looks like: dtv-type-spec, INTENT(INOUT) :: dtv
  if (CheckDioDummyIsData(subp, arg, 0)) {
    CheckDioDummyIsDerived(subp, *arg, ioKind);
    CheckDioDummyAttrs(subp, *arg,
        ioKind == GenericKind::DefinedIo::ReadFormatted ||
                ioKind == GenericKind::DefinedIo::ReadUnformatted
            ? Attr::INTENT_INOUT
            : Attr::INTENT_IN);
  }
}

// If an explicit INTRINSIC name is a function, so must all the specifics be,
// and similarly for subroutines
void CheckHelper::CheckGenericVsIntrinsic(
    const Symbol &symbol, const GenericDetails &generic) {
  if (symbol.attrs().test(Attr::INTRINSIC)) {
    const evaluate::IntrinsicProcTable &table{
        context_.foldingContext().intrinsics()};
    bool isSubroutine{table.IsIntrinsicSubroutine(symbol.name().ToString())};
    if (isSubroutine || table.IsIntrinsicFunction(symbol.name().ToString())) {
      for (const SymbolRef &ref : generic.specificProcs()) {
        const Symbol &ultimate{ref->GetUltimate()};
        bool specificFunc{ultimate.test(Symbol::Flag::Function)};
        bool specificSubr{ultimate.test(Symbol::Flag::Subroutine)};
        if (!specificFunc && !specificSubr) {
          if (const auto *proc{ultimate.detailsIf<SubprogramDetails>()}) {
            if (proc->isFunction()) {
              specificFunc = true;
            } else {
              specificSubr = true;
            }
          }
        }
        if ((specificFunc || specificSubr) &&
            isSubroutine != specificSubr) { // C848
          messages_.Say(symbol.name(),
              "Generic interface '%s' with explicit intrinsic %s of the same name may not have specific procedure '%s' that is a %s"_err_en_US,
              symbol.name(), isSubroutine ? "subroutine" : "function",
              ref->name(), isSubroutine ? "function" : "subroutine");
        }
      }
    }
  }
}

void CheckHelper::CheckDefaultIntegerArg(
    const Symbol &subp, const Symbol *arg, Attr intent) {
  // Argument looks like: INTEGER, INTENT(intent) :: arg
  if (CheckDioDummyIsData(subp, arg, 1)) {
    CheckDioDummyIsDefaultInteger(subp, *arg);
    CheckDioDummyIsScalar(subp, *arg);
    CheckDioDummyAttrs(subp, *arg, intent);
  }
}

void CheckHelper::CheckDioAssumedLenCharacterArg(const Symbol &subp,
    const Symbol *arg, std::size_t argPosition, Attr intent) {
  // Argument looks like: CHARACTER (LEN=*), INTENT(intent) :: (iotype OR iomsg)
  if (CheckDioDummyIsData(subp, arg, argPosition)) {
    CheckDioDummyAttrs(subp, *arg, intent);
    if (!IsAssumedLengthCharacter(*arg)) {
      messages_.Say(arg->name(),
          "Dummy argument '%s' of a defined input/output procedure"
          " must be assumed-length CHARACTER"_err_en_US,
          arg->name());
    }
  }
}

void CheckHelper::CheckDioVlistArg(
    const Symbol &subp, const Symbol *arg, std::size_t argPosition) {
  // Vlist argument looks like: INTEGER, INTENT(IN) :: v_list(:)
  if (CheckDioDummyIsData(subp, arg, argPosition)) {
    CheckDioDummyIsDefaultInteger(subp, *arg);
    CheckDioDummyAttrs(subp, *arg, Attr::INTENT_IN);
    const auto *objectDetails{arg->detailsIf<ObjectEntityDetails>()};
    if (!objectDetails || !objectDetails->shape().CanBeDeferredShape()) {
      messages_.Say(arg->name(),
          "Dummy argument '%s' of a defined input/output procedure must be"
          " deferred shape"_err_en_US,
          arg->name());
    }
  }
}

void CheckHelper::CheckDioArgCount(
    const Symbol &subp, GenericKind::DefinedIo ioKind, std::size_t argCount) {
  const std::size_t requiredArgCount{
      (std::size_t)(ioKind == GenericKind::DefinedIo::ReadFormatted ||
                  ioKind == GenericKind::DefinedIo::WriteFormatted
              ? 6
              : 4)};
  if (argCount != requiredArgCount) {
    SayWithDeclaration(subp,
        "Defined input/output procedure '%s' must have"
        " %d dummy arguments rather than %d"_err_en_US,
        subp.name(), requiredArgCount, argCount);
    context_.SetError(subp);
  }
}

void CheckHelper::CheckDioDummyAttrs(
    const Symbol &subp, const Symbol &arg, Attr goodIntent) {
  // Defined I/O procedures can't have attributes other than INTENT
  Attrs attrs{arg.attrs()};
  if (!attrs.test(goodIntent)) {
    messages_.Say(arg.name(),
        "Dummy argument '%s' of a defined input/output procedure"
        " must have intent '%s'"_err_en_US,
        arg.name(), AttrToString(goodIntent));
  }
  attrs = attrs - Attr::INTENT_IN - Attr::INTENT_OUT - Attr::INTENT_INOUT;
  if (!attrs.empty()) {
    messages_.Say(arg.name(),
        "Dummy argument '%s' of a defined input/output procedure may not have"
        " any attributes"_err_en_US,
        arg.name());
  }
}

// Enforce semantics for defined input/output procedures (12.6.4.8.2) and C777
void CheckHelper::CheckDefinedIoProc(const Symbol &symbol,
    const GenericDetails &details, GenericKind::DefinedIo ioKind) {
  for (auto ref : details.specificProcs()) {
    const auto *binding{ref->detailsIf<ProcBindingDetails>()};
    const Symbol &specific{*(binding ? &binding->symbol() : &*ref)};
    if (ref->attrs().test(Attr::NOPASS)) { // C774
      messages_.Say("Defined input/output procedure '%s' may not have NOPASS "
                    "attribute"_err_en_US,
          ref->name());
      context_.SetError(*ref);
    }
    if (const auto *subpDetails{specific.detailsIf<SubprogramDetails>()}) {
      const std::vector<Symbol *> &dummyArgs{subpDetails->dummyArgs()};
      CheckDioArgCount(specific, ioKind, dummyArgs.size());
      int argCount{0};
      for (auto *arg : dummyArgs) {
        switch (argCount++) {
        case 0:
          // dtv-type-spec, INTENT(INOUT) :: dtv
          CheckDioDtvArg(specific, arg, ioKind);
          break;
        case 1:
          // INTEGER, INTENT(IN) :: unit
          CheckDefaultIntegerArg(specific, arg, Attr::INTENT_IN);
          break;
        case 2:
          if (ioKind == GenericKind::DefinedIo::ReadFormatted ||
              ioKind == GenericKind::DefinedIo::WriteFormatted) {
            // CHARACTER (LEN=*), INTENT(IN) :: iotype
            CheckDioAssumedLenCharacterArg(
                specific, arg, argCount, Attr::INTENT_IN);
          } else {
            // INTEGER, INTENT(OUT) :: iostat
            CheckDefaultIntegerArg(specific, arg, Attr::INTENT_OUT);
          }
          break;
        case 3:
          if (ioKind == GenericKind::DefinedIo::ReadFormatted ||
              ioKind == GenericKind::DefinedIo::WriteFormatted) {
            // INTEGER, INTENT(IN) :: v_list(:)
            CheckDioVlistArg(specific, arg, argCount);
          } else {
            // CHARACTER (LEN=*), INTENT(INOUT) :: iomsg
            CheckDioAssumedLenCharacterArg(
                specific, arg, argCount, Attr::INTENT_INOUT);
          }
          break;
        case 4:
          // INTEGER, INTENT(OUT) :: iostat
          CheckDefaultIntegerArg(specific, arg, Attr::INTENT_OUT);
          break;
        case 5:
          // CHARACTER (LEN=*), INTENT(INOUT) :: iomsg
          CheckDioAssumedLenCharacterArg(
              specific, arg, argCount, Attr::INTENT_INOUT);
          break;
        default:;
        }
      }
    }
  }
}

void SubprogramMatchHelper::Check(
    const Symbol &symbol1, const Symbol &symbol2) {
  const auto details1{symbol1.get<SubprogramDetails>()};
  const auto details2{symbol2.get<SubprogramDetails>()};
  if (details1.isFunction() != details2.isFunction()) {
    Say(symbol1, symbol2,
        details1.isFunction()
            ? "Module function '%s' was declared as a subroutine in the"
              " corresponding interface body"_err_en_US
            : "Module subroutine '%s' was declared as a function in the"
              " corresponding interface body"_err_en_US);
    return;
  }
  const auto &args1{details1.dummyArgs()};
  const auto &args2{details2.dummyArgs()};
  int nargs1{static_cast<int>(args1.size())};
  int nargs2{static_cast<int>(args2.size())};
  if (nargs1 != nargs2) {
    Say(symbol1, symbol2,
        "Module subprogram '%s' has %d args but the corresponding interface"
        " body has %d"_err_en_US,
        nargs1, nargs2);
    return;
  }
  bool nonRecursive1{symbol1.attrs().test(Attr::NON_RECURSIVE)};
  if (nonRecursive1 != symbol2.attrs().test(Attr::NON_RECURSIVE)) { // C1551
    Say(symbol1, symbol2,
        nonRecursive1
            ? "Module subprogram '%s' has NON_RECURSIVE prefix but"
              " the corresponding interface body does not"_err_en_US
            : "Module subprogram '%s' does not have NON_RECURSIVE prefix but "
              "the corresponding interface body does"_err_en_US);
  }
  const std::string *bindName1{details1.bindName()};
  const std::string *bindName2{details2.bindName()};
  if (!bindName1 && !bindName2) {
    // OK - neither has a binding label
  } else if (!bindName1) {
    Say(symbol1, symbol2,
        "Module subprogram '%s' does not have a binding label but the"
        " corresponding interface body does"_err_en_US);
  } else if (!bindName2) {
    Say(symbol1, symbol2,
        "Module subprogram '%s' has a binding label but the"
        " corresponding interface body does not"_err_en_US);
  } else if (*bindName1 != *bindName2) {
    Say(symbol1, symbol2,
        "Module subprogram '%s' has binding label '%s' but the corresponding"
        " interface body has '%s'"_err_en_US,
        *details1.bindName(), *details2.bindName());
  }
  const Procedure *proc1{checkHelper.Characterize(symbol1)};
  const Procedure *proc2{checkHelper.Characterize(symbol2)};
  if (!proc1 || !proc2) {
    return;
  }
  if (proc1->attrs.test(Procedure::Attr::Pure) !=
      proc2->attrs.test(Procedure::Attr::Pure)) {
    Say(symbol1, symbol2,
        "Module subprogram '%s' and its corresponding interface body are not both PURE"_err_en_US);
  }
  if (proc1->attrs.test(Procedure::Attr::Elemental) !=
      proc2->attrs.test(Procedure::Attr::Elemental)) {
    Say(symbol1, symbol2,
        "Module subprogram '%s' and its corresponding interface body are not both ELEMENTAL"_err_en_US);
  }
  if (proc1->attrs.test(Procedure::Attr::BindC) !=
      proc2->attrs.test(Procedure::Attr::BindC)) {
    Say(symbol1, symbol2,
        "Module subprogram '%s' and its corresponding interface body are not both BIND(C)"_err_en_US);
  }
  if (proc1->functionResult && proc2->functionResult &&
      *proc1->functionResult != *proc2->functionResult) {
    Say(symbol1, symbol2,
        "Return type of function '%s' does not match return type of"
        " the corresponding interface body"_err_en_US);
  }
  for (int i{0}; i < nargs1; ++i) {
    const Symbol *arg1{args1[i]};
    const Symbol *arg2{args2[i]};
    if (arg1 && !arg2) {
      Say(symbol1, symbol2,
          "Dummy argument %2$d of '%1$s' is not an alternate return indicator"
          " but the corresponding argument in the interface body is"_err_en_US,
          i + 1);
    } else if (!arg1 && arg2) {
      Say(symbol1, symbol2,
          "Dummy argument %2$d of '%1$s' is an alternate return indicator but"
          " the corresponding argument in the interface body is not"_err_en_US,
          i + 1);
    } else if (arg1 && arg2) {
      SourceName name1{arg1->name()};
      SourceName name2{arg2->name()};
      if (name1 != name2) {
        Say(*arg1, *arg2,
            "Dummy argument name '%s' does not match corresponding name '%s'"
            " in interface body"_err_en_US,
            name2);
      } else {
        CheckDummyArg(
            *arg1, *arg2, proc1->dummyArguments[i], proc2->dummyArguments[i]);
      }
    }
  }
}

void SubprogramMatchHelper::CheckDummyArg(const Symbol &symbol1,
    const Symbol &symbol2, const DummyArgument &arg1,
    const DummyArgument &arg2) {
  common::visit(
      common::visitors{
          [&](const DummyDataObject &obj1, const DummyDataObject &obj2) {
            CheckDummyDataObject(symbol1, symbol2, obj1, obj2);
          },
          [&](const DummyProcedure &proc1, const DummyProcedure &proc2) {
            CheckDummyProcedure(symbol1, symbol2, proc1, proc2);
          },
          [&](const DummyDataObject &, const auto &) {
            Say(symbol1, symbol2,
                "Dummy argument '%s' is a data object; the corresponding"
                " argument in the interface body is not"_err_en_US);
          },
          [&](const DummyProcedure &, const auto &) {
            Say(symbol1, symbol2,
                "Dummy argument '%s' is a procedure; the corresponding"
                " argument in the interface body is not"_err_en_US);
          },
          [&](const auto &, const auto &) {
            llvm_unreachable("Dummy arguments are not data objects or"
                             "procedures");
          },
      },
      arg1.u, arg2.u);
}

void SubprogramMatchHelper::CheckDummyDataObject(const Symbol &symbol1,
    const Symbol &symbol2, const DummyDataObject &obj1,
    const DummyDataObject &obj2) {
  if (!CheckSameIntent(symbol1, symbol2, obj1.intent, obj2.intent)) {
  } else if (!CheckSameAttrs(symbol1, symbol2, obj1.attrs, obj2.attrs)) {
  } else if (obj1.type.type() != obj2.type.type()) {
    Say(symbol1, symbol2,
        "Dummy argument '%s' has type %s; the corresponding argument in the"
        " interface body has type %s"_err_en_US,
        obj1.type.type().AsFortran(), obj2.type.type().AsFortran());
  } else if (!ShapesAreCompatible(obj1, obj2)) {
    Say(symbol1, symbol2,
        "The shape of dummy argument '%s' does not match the shape of the"
        " corresponding argument in the interface body"_err_en_US);
  }
  // TODO: coshape
}

void SubprogramMatchHelper::CheckDummyProcedure(const Symbol &symbol1,
    const Symbol &symbol2, const DummyProcedure &proc1,
    const DummyProcedure &proc2) {
  if (!CheckSameIntent(symbol1, symbol2, proc1.intent, proc2.intent)) {
  } else if (!CheckSameAttrs(symbol1, symbol2, proc1.attrs, proc2.attrs)) {
  } else if (proc1 != proc2) {
    Say(symbol1, symbol2,
        "Dummy procedure '%s' does not match the corresponding argument in"
        " the interface body"_err_en_US);
  }
}

bool SubprogramMatchHelper::CheckSameIntent(const Symbol &symbol1,
    const Symbol &symbol2, common::Intent intent1, common::Intent intent2) {
  if (intent1 == intent2) {
    return true;
  } else {
    Say(symbol1, symbol2,
        "The intent of dummy argument '%s' does not match the intent"
        " of the corresponding argument in the interface body"_err_en_US);
    return false;
  }
}

// Report an error referring to first symbol with declaration of second symbol
template <typename... A>
void SubprogramMatchHelper::Say(const Symbol &symbol1, const Symbol &symbol2,
    parser::MessageFixedText &&text, A &&...args) {
  auto &message{context().Say(symbol1.name(), std::move(text), symbol1.name(),
      std::forward<A>(args)...)};
  evaluate::AttachDeclaration(message, symbol2);
}

template <typename ATTRS>
bool SubprogramMatchHelper::CheckSameAttrs(
    const Symbol &symbol1, const Symbol &symbol2, ATTRS attrs1, ATTRS attrs2) {
  if (attrs1 == attrs2) {
    return true;
  }
  attrs1.IterateOverMembers([&](auto attr) {
    if (!attrs2.test(attr)) {
      Say(symbol1, symbol2,
          "Dummy argument '%s' has the %s attribute; the corresponding"
          " argument in the interface body does not"_err_en_US,
          AsFortran(attr));
    }
  });
  attrs2.IterateOverMembers([&](auto attr) {
    if (!attrs1.test(attr)) {
      Say(symbol1, symbol2,
          "Dummy argument '%s' does not have the %s attribute; the"
          " corresponding argument in the interface body does"_err_en_US,
          AsFortran(attr));
    }
  });
  return false;
}

bool SubprogramMatchHelper::ShapesAreCompatible(
    const DummyDataObject &obj1, const DummyDataObject &obj2) {
  return characteristics::ShapesAreCompatible(
      FoldShape(obj1.type.shape()), FoldShape(obj2.type.shape()));
}

evaluate::Shape SubprogramMatchHelper::FoldShape(const evaluate::Shape &shape) {
  evaluate::Shape result;
  for (const auto &extent : shape) {
    result.emplace_back(
        evaluate::Fold(context().foldingContext(), common::Clone(extent)));
  }
  return result;
}

void DistinguishabilityHelper::Add(const Symbol &generic, GenericKind kind,
    const Symbol &specific, const Procedure &procedure) {
  if (!context_.HasError(specific)) {
    nameToInfo_[generic.name()].emplace_back(
        ProcedureInfo{kind, specific, procedure});
  }
}

void DistinguishabilityHelper::Check(const Scope &scope) {
  for (const auto &[name, info] : nameToInfo_) {
    auto count{info.size()};
    for (std::size_t i1{0}; i1 < count - 1; ++i1) {
      const auto &[kind, symbol, proc]{info[i1]};
      for (std::size_t i2{i1 + 1}; i2 < count; ++i2) {
        auto distinguishable{kind.IsName()
                ? evaluate::characteristics::Distinguishable
                : evaluate::characteristics::DistinguishableOpOrAssign};
        if (!distinguishable(
                context_.languageFeatures(), proc, info[i2].procedure)) {
          SayNotDistinguishable(GetTopLevelUnitContaining(scope), name, kind,
              symbol, info[i2].symbol);
        }
      }
    }
  }
}

void DistinguishabilityHelper::SayNotDistinguishable(const Scope &scope,
    const SourceName &name, GenericKind kind, const Symbol &proc1,
    const Symbol &proc2) {
  std::string name1{proc1.name().ToString()};
  std::string name2{proc2.name().ToString()};
  if (kind.IsOperator() || kind.IsAssignment()) {
    // proc1 and proc2 may come from different scopes so qualify their names
    if (proc1.owner().IsDerivedType()) {
      name1 = proc1.owner().GetName()->ToString() + '%' + name1;
    }
    if (proc2.owner().IsDerivedType()) {
      name2 = proc2.owner().GetName()->ToString() + '%' + name2;
    }
  }
  parser::Message *msg;
  if (scope.sourceRange().Contains(name)) {
    msg = &context_.Say(name,
        "Generic '%s' may not have specific procedures '%s' and '%s' as their interfaces are not distinguishable"_err_en_US,
        MakeOpName(name), name1, name2);
  } else {
    msg = &context_.Say(*GetTopLevelUnitContaining(proc1).GetName(),
        "USE-associated generic '%s' may not have specific procedures '%s' and '%s' as their interfaces are not distinguishable"_err_en_US,
        MakeOpName(name), name1, name2);
  }
  AttachDeclaration(*msg, scope, proc1);
  AttachDeclaration(*msg, scope, proc2);
}

// `evaluate::AttachDeclaration` doesn't handle the generic case where `proc`
// comes from a different module but is not necessarily use-associated.
void DistinguishabilityHelper::AttachDeclaration(
    parser::Message &msg, const Scope &scope, const Symbol &proc) {
  const Scope &unit{GetTopLevelUnitContaining(proc)};
  if (unit == scope) {
    evaluate::AttachDeclaration(msg, proc);
  } else {
    msg.Attach(unit.GetName().value(),
        "'%s' is USE-associated from module '%s'"_en_US, proc.name(),
        unit.GetName().value());
  }
}

void CheckDeclarations(SemanticsContext &context) {
  CheckHelper{context}.Check();
}
} // namespace Fortran::semantics
