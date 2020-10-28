//===-- lib/Semantics/check-declarations.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Static declaration checking

#include "check-declarations.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/fold.h"
#include "flang/Evaluate/tools.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include <algorithm>

namespace Fortran::semantics {

using evaluate::characteristics::DummyArgument;
using evaluate::characteristics::DummyDataObject;
using evaluate::characteristics::DummyProcedure;
using evaluate::characteristics::FunctionResult;
using evaluate::characteristics::Procedure;

class CheckHelper {
public:
  explicit CheckHelper(SemanticsContext &c) : context_{c} {}

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

private:
  template <typename A> void CheckSpecExpr(const A &x) {
    evaluate::CheckSpecificationExpr(
        x, messages_, DEREF(scope_), context_.intrinsics());
  }
  void CheckValue(const Symbol &, const DerivedTypeSpec *);
  void CheckVolatile(
      const Symbol &, bool isAssociated, const DerivedTypeSpec *);
  void CheckPointer(const Symbol &);
  void CheckPassArg(
      const Symbol &proc, const Symbol *interface, const WithPassArg &);
  void CheckProcBinding(const Symbol &, const ProcBindingDetails &);
  void CheckObjectEntity(const Symbol &, const ObjectEntityDetails &);
  void CheckArraySpec(const Symbol &, const ArraySpec &);
  void CheckProcEntity(const Symbol &, const ProcEntityDetails &);
  void CheckSubprogram(const Symbol &, const SubprogramDetails &);
  void CheckAssumedTypeEntity(const Symbol &, const ObjectEntityDetails &);
  void CheckDerivedType(const Symbol &, const DerivedTypeDetails &);
  void CheckGeneric(const Symbol &, const GenericDetails &);
  std::optional<std::vector<Procedure>> Characterize(const SymbolVector &);
  bool CheckDefinedOperator(const SourceName &, const GenericKind &,
      const Symbol &, const Procedure &);
  std::optional<parser::MessageFixedText> CheckNumberOfArgs(
      const GenericKind &, std::size_t);
  bool CheckDefinedOperatorArg(
      const SourceName &, const Symbol &, const Procedure &, std::size_t);
  bool CheckDefinedAssignment(const Symbol &, const Procedure &);
  bool CheckDefinedAssignmentArg(const Symbol &, const DummyArgument &, int);
  void CheckSpecificsAreDistinguishable(
      const Symbol &, const GenericDetails &, const std::vector<Procedure> &);
  void CheckEquivalenceSet(const EquivalenceSet &);
  void CheckBlockData(const Scope &);

  void SayNotDistinguishable(
      const SourceName &, GenericKind, const Symbol &, const Symbol &);
  bool CheckConflicting(const Symbol &, Attr, Attr);
  bool InPure() const {
    return innermostSymbol_ && IsPureProcedure(*innermostSymbol_);
  }
  bool InFunction() const {
    return innermostSymbol_ && IsFunction(*innermostSymbol_);
  }
  template <typename... A>
  void SayWithDeclaration(const Symbol &symbol, A &&... x) {
    if (parser::Message * msg{messages_.Say(std::forward<A>(x)...)}) {
      if (messages_.at().begin() != symbol.name().begin()) {
        evaluate::AttachDeclaration(*msg, symbol);
      }
    }
  }
  bool IsResultOkToDiffer(const FunctionResult &);

  SemanticsContext &context_;
  evaluate::FoldingContext &foldingContext_{context_.foldingContext()};
  parser::ContextualMessages &messages_{foldingContext_.messages()};
  const Scope *scope_{nullptr};
  // This symbol is the one attached to the innermost enclosing scope
  // that has a symbol.
  const Symbol *innermostSymbol_{nullptr};
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
  if (context_.HasError(symbol)) {
    return;
  }
  const DeclTypeSpec *type{symbol.GetType()};
  const DerivedTypeSpec *derived{type ? type->AsDerived() : nullptr};
  auto restorer{messages_.SetLocation(symbol.name())};
  context_.set_location(symbol.name());
  bool isAssociated{symbol.has<UseDetails>() || symbol.has<HostAssocDetails>()};
  if (symbol.attrs().test(Attr::VOLATILE)) {
    CheckVolatile(symbol, isAssociated, derived);
  }
  if (isAssociated) {
    return; // only care about checking VOLATILE on associated symbols
  }
  if (IsPointer(symbol)) {
    CheckPointer(symbol);
  }
  std::visit(
      common::visitors{
          [&](const ProcBindingDetails &x) { CheckProcBinding(symbol, x); },
          [&](const ObjectEntityDetails &x) { CheckObjectEntity(symbol, x); },
          [&](const ProcEntityDetails &x) { CheckProcEntity(symbol, x); },
          [&](const SubprogramDetails &x) { CheckSubprogram(symbol, x); },
          [&](const DerivedTypeDetails &x) { CheckDerivedType(symbol, x); },
          [&](const GenericDetails &x) { CheckGeneric(symbol, x); },
          [](const auto &) {},
      },
      symbol.details());
  if (InPure()) {
    if (IsSaved(symbol)) {
      messages_.Say(
          "A pure subprogram may not have a variable with the SAVE attribute"_err_en_US);
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
    if (IsSaved(symbol)) {
      messages_.Say(
          "A dummy argument may not have the SAVE attribute"_err_en_US);
    }
  } else if (IsFunctionResult(symbol)) {
    if (IsSaved(symbol)) {
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
  if (IsCoarray(symbol)) {
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
      if (IsCoarray(symbol)) {
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
  CheckAssumedTypeEntity(symbol, details);
  if (!details.coshape().empty()) {
    bool isDeferredShape{details.coshape().IsDeferredShape()};
    if (IsAllocatable(symbol)) {
      if (!isDeferredShape) { // C827
        messages_.Say("'%s' is an ALLOCATABLE coarray and must have a deferred"
                      " coshape"_err_en_US,
            symbol.name());
      }
    } else if (symbol.owner().IsDerivedType()) { // C746
      std::string deferredMsg{
          isDeferredShape ? "" : " and have a deferred coshape"};
      messages_.Say("Component '%s' is a coarray and must have the ALLOCATABLE"
                    " attribute%s"_err_en_US,
          symbol.name(), deferredMsg);
    } else {
      if (!details.coshape().IsAssumedSize()) { // C828
        messages_.Say(
            "Component '%s' is a non-ALLOCATABLE coarray and must have"
            " an explicit coshape"_err_en_US,
            symbol.name());
      }
    }
  }
  if (details.isDummy()) {
    if (symbol.attrs().test(Attr::INTENT_OUT)) {
      if (FindUltimateComponent(symbol, [](const Symbol &x) {
            return IsCoarray(x) && IsAllocatable(x);
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
  }
  if (symbol.owner().kind() != Scope::Kind::DerivedType &&
      IsInitialized(symbol, true /*ignore DATA, already caught*/)) { // C808
    if (IsAutomatic(symbol)) {
      messages_.Say("An automatic variable must not be initialized"_err_en_US);
    } else if (IsDummy(symbol)) {
      messages_.Say("A dummy argument must not be initialized"_err_en_US);
    } else if (IsFunctionResult(symbol)) {
      messages_.Say("A function result must not be initialized"_err_en_US);
    } else if (IsInBlankCommon(symbol)) {
      messages_.Say(
          "A variable in blank COMMON should not be initialized"_en_US);
    }
  }
  if (symbol.owner().kind() == Scope::Kind::BlockData &&
      IsInitialized(symbol)) {
    if (IsAllocatable(symbol)) {
      messages_.Say(
          "An ALLOCATABLE variable may not appear in a BLOCK DATA subprogram"_err_en_US);
    } else if (!FindCommonBlockContaining(symbol)) {
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
  bool isDeferred{arraySpec.IsDeferredShape()};
  bool isImplied{arraySpec.IsImpliedShape()};
  bool isAssumedShape{arraySpec.IsAssumedShape()};
  bool isAssumedSize{arraySpec.IsAssumedSize()};
  bool isAssumedRank{arraySpec.IsAssumedRank()};
  std::optional<parser::MessageFixedText> msg;
  if (symbol.test(Symbol::Flag::CrayPointee) && !isExplicit && !isAssumedSize) {
    msg = "Cray pointee '%s' must have must have explicit shape or"
          " assumed size"_err_en_US;
  } else if (IsAllocatableOrPointer(symbol) && !isDeferred && !isAssumedRank) {
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
    if (isImplied && !isAssumedSize) { // C836
      msg = "Dummy array argument '%s' may not have implied shape"_err_en_US;
    }
  } else if (isAssumedShape && !isDeferred) {
    msg = "Assumed-shape array '%s' must be a dummy argument"_err_en_US;
  } else if (isAssumedSize && !isImplied) { // C833
    msg = "Assumed-size array '%s' must be a dummy argument"_err_en_US;
  } else if (isAssumedRank) { // C837
    msg = "Assumed-rank array '%s' must be a dummy argument"_err_en_US;
  } else if (isImplied) {
    if (!IsNamedConstant(symbol)) { // C836
      msg = "Implied-shape array '%s' must be a named constant"_err_en_US;
    }
  } else if (IsNamedConstant(symbol)) {
    if (!isExplicit && !isImplied) {
      msg = "Named constant '%s' array must have explicit or"
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
    const Symbol *interface{details.interface().symbol()};
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
    if (const Symbol * interface{details.interface().symbol()}) {
      if (interface->attrs().test(Attr::ELEMENTAL) &&
          !interface->attrs().test(Attr::INTRINSIC)) {
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
  explicit SubprogramMatchHelper(SemanticsContext &context)
      : context{context} {}

  void Check(const Symbol &, const Symbol &);

private:
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

  SemanticsContext &context;
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
    SubprogramMatchHelper{context_}.Check(symbol, *iface);
  }
  if (const Scope * entryScope{details.entryScope()}) {
    // ENTRY 15.6.2.6, esp. C1571
    std::optional<parser::MessageFixedText> error;
    const Symbol *subprogram{entryScope->symbol()};
    const SubprogramDetails *subprogramDetails{nullptr};
    if (subprogram) {
      subprogramDetails = subprogram->detailsIf<SubprogramDetails>();
    }
    if (entryScope->kind() != Scope::Kind::Subprogram) {
      error = "ENTRY may appear only in a subroutine or function"_err_en_US;
    } else if (!(entryScope->parent().IsGlobal() ||
                   entryScope->parent().IsModule() ||
                   entryScope->parent().IsSubmodule())) {
      error = "ENTRY may not appear in an internal subprogram"_err_en_US;
    } else if (FindSeparateModuleSubprogramInterface(subprogram)) {
      error = "ENTRY may not appear in a separate module procedure"_err_en_US;
    } else if (subprogramDetails && details.isFunction() &&
        subprogramDetails->isFunction()) {
      auto result{FunctionResult::Characterize(
          details.result(), context_.intrinsics())};
      auto subpResult{FunctionResult::Characterize(
          subprogramDetails->result(), context_.intrinsics())};
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
}

void CheckHelper::CheckDerivedType(
    const Symbol &symbol, const DerivedTypeDetails &details) {
  const Scope *scope{symbol.scope()};
  if (!scope) {
    CHECK(details.isForwardReferenced());
    return;
  }
  CHECK(scope->symbol() == &symbol);
  CHECK(scope->IsDerivedType());
  if (symbol.attrs().test(Attr::ABSTRACT) && // C734
      (symbol.attrs().test(Attr::BIND_C) || details.sequence())) {
    messages_.Say("An ABSTRACT derived type must be extensible"_err_en_US);
  }
  if (const DeclTypeSpec * parent{FindParentTypeSpec(symbol)}) {
    const DerivedTypeSpec *parentDerived{parent->AsDerived()};
    if (!IsExtensibleType(parentDerived)) { // C705
      messages_.Say("The parent type is not extensible"_err_en_US);
    }
    if (!symbol.attrs().test(Attr::ABSTRACT) && parentDerived &&
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
    DerivedTypeSpec derived{symbol.name(), symbol};
    derived.set_scope(*scope);
    if (FindCoarrayUltimateComponent(derived) && // C736
        !(parentDerived && FindCoarrayUltimateComponent(*parentDerived))) {
      messages_.Say(
          "Type '%s' has a coarray ultimate component so the type at the base "
          "of its type extension chain ('%s') must be a type that has a "
          "coarray ultimate component"_err_en_US,
          symbol.name(), scope->GetDerivedTypeBase().GetSymbol()->name());
    }
    if (FindEventOrLockPotentialComponent(derived) && // C737
        !(FindEventOrLockPotentialComponent(*parentDerived) ||
            IsEventTypeOrLockType(parentDerived))) {
      messages_.Say(
          "Type '%s' has an EVENT_TYPE or LOCK_TYPE component, so the type "
          "at the base of its type extension chain ('%s') must either have an "
          "EVENT_TYPE or LOCK_TYPE component, or be EVENT_TYPE or "
          "LOCK_TYPE"_err_en_US,
          symbol.name(), scope->GetDerivedTypeBase().GetSymbol()->name());
    }
  }
  if (HasIntrinsicTypeName(symbol)) { // C729
    messages_.Say("A derived type name cannot be the name of an intrinsic"
                  " type"_err_en_US);
  }
}

void CheckHelper::CheckGeneric(
    const Symbol &symbol, const GenericDetails &details) {
  const SymbolVector &specifics{details.specificProcs()};
  const auto &bindingNames{details.bindingNames()};
  std::optional<std::vector<Procedure>> procs{Characterize(specifics)};
  if (!procs) {
    return;
  }
  bool ok{true};
  if (details.kind().IsIntrinsicOperator()) {
    for (std::size_t i{0}; i < specifics.size(); ++i) {
      auto restorer{messages_.SetLocation(bindingNames[i])};
      ok &= CheckDefinedOperator(
          symbol.name(), details.kind(), specifics[i], (*procs)[i]);
    }
  }
  if (details.kind().IsAssignment()) {
    for (std::size_t i{0}; i < specifics.size(); ++i) {
      auto restorer{messages_.SetLocation(bindingNames[i])};
      ok &= CheckDefinedAssignment(specifics[i], (*procs)[i]);
    }
  }
  if (ok) {
    CheckSpecificsAreDistinguishable(symbol, details, *procs);
  }
}

// Check that the specifics of this generic are distinguishable from each other
void CheckHelper::CheckSpecificsAreDistinguishable(const Symbol &generic,
    const GenericDetails &details, const std::vector<Procedure> &procs) {
  const SymbolVector &specifics{details.specificProcs()};
  std::size_t count{specifics.size()};
  if (count < 2) {
    return;
  }
  GenericKind kind{details.kind()};
  auto distinguishable{kind.IsAssignment() || kind.IsOperator()
          ? evaluate::characteristics::DistinguishableOpOrAssign
          : evaluate::characteristics::Distinguishable};
  for (std::size_t i1{0}; i1 < count - 1; ++i1) {
    auto &proc1{procs[i1]};
    for (std::size_t i2{i1 + 1}; i2 < count; ++i2) {
      auto &proc2{procs[i2]};
      if (!distinguishable(proc1, proc2)) {
        SayNotDistinguishable(
            generic.name(), kind, specifics[i1], specifics[i2]);
      }
    }
  }
}

void CheckHelper::SayNotDistinguishable(const SourceName &name,
    GenericKind kind, const Symbol &proc1, const Symbol &proc2) {
  auto &&text{kind.IsDefinedOperator()
          ? "Generic operator '%s' may not have specific procedures '%s'"
            " and '%s' as their interfaces are not distinguishable"_err_en_US
          : "Generic '%s' may not have specific procedures '%s'"
            " and '%s' as their interfaces are not distinguishable"_err_en_US};
  auto &msg{
      context_.Say(name, std::move(text), name, proc1.name(), proc2.name())};
  evaluate::AttachDeclaration(msg, proc1);
  evaluate::AttachDeclaration(msg, proc2);
}

static bool ConflictsWithIntrinsicAssignment(const Procedure &proc) {
  auto lhs{std::get<DummyDataObject>(proc.dummyArguments[0].u).type};
  auto rhs{std::get<DummyDataObject>(proc.dummyArguments[1].u).type};
  return Tristate::No ==
      IsDefinedAssignment(lhs.type(), lhs.Rank(), rhs.type(), rhs.Rank());
}

static bool ConflictsWithIntrinsicOperator(
    const GenericKind &kind, const Procedure &proc) {
  auto arg0{std::get<DummyDataObject>(proc.dummyArguments[0].u).type};
  auto type0{arg0.type()};
  if (proc.dummyArguments.size() == 1) { // unary
    return std::visit(
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
    return std::visit(
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
bool CheckHelper::CheckDefinedOperator(const SourceName &opName,
    const GenericKind &kind, const Symbol &specific, const Procedure &proc) {
  std::optional<parser::MessageFixedText> msg;
  if (specific.attrs().test(Attr::NOPASS)) { // C774
    msg = "%s procedure '%s' may not have NOPASS attribute"_err_en_US;
  } else if (!proc.functionResult.has_value()) {
    msg = "%s procedure '%s' must be a function"_err_en_US;
  } else if (proc.functionResult->IsAssumedLengthCharacter()) {
    msg = "%s function '%s' may not have assumed-length CHARACTER(*)"
          " result"_err_en_US;
  } else if (auto m{CheckNumberOfArgs(kind, proc.dummyArguments.size())}) {
    msg = std::move(m);
  } else if (!CheckDefinedOperatorArg(opName, specific, proc, 0) |
      !CheckDefinedOperatorArg(opName, specific, proc, 1)) {
    return false; // error was reported
  } else if (ConflictsWithIntrinsicOperator(kind, proc)) {
    msg = "%s function '%s' conflicts with intrinsic operator"_err_en_US;
  } else {
    return true; // OK
  }
  SayWithDeclaration(specific, std::move(msg.value()),
      parser::ToUpperCaseLetters(opName.ToString()), specific.name());
  return false;
}

// If the number of arguments is wrong for this intrinsic operator, return
// false and return the error message in msg.
std::optional<parser::MessageFixedText> CheckHelper::CheckNumberOfArgs(
    const GenericKind &kind, std::size_t nargs) {
  std::size_t min{2}, max{2}; // allowed number of args; default is binary
  std::visit(common::visitors{
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
  std::optional<parser::MessageFixedText> msg;
  if (specific.attrs().test(Attr::NOPASS)) { // C774
    msg = "Defined assignment procedure '%s' may not have"
          " NOPASS attribute"_err_en_US;
  } else if (!proc.IsSubroutine()) {
    msg = "Defined assignment procedure '%s' must be a subroutine"_err_en_US;
  } else if (proc.dummyArguments.size() != 2) {
    msg = "Defined assignment subroutine '%s' must have"
          " two dummy arguments"_err_en_US;
  } else if (!CheckDefinedAssignmentArg(specific, proc.dummyArguments[0], 0) |
      !CheckDefinedAssignmentArg(specific, proc.dummyArguments[1], 1)) {
    return false; // error was reported
  } else if (ConflictsWithIntrinsicAssignment(proc)) {
    msg = "Defined assignment subroutine '%s' conflicts with"
          " intrinsic assignment"_err_en_US;
  } else {
    return true; // OK
  }
  SayWithDeclaration(specific, std::move(msg.value()), specific.name());
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
    return false;
  }
  return true;
}

// Report a conflicting attribute error if symbol has both of these attributes
bool CheckHelper::CheckConflicting(const Symbol &symbol, Attr a1, Attr a2) {
  if (symbol.attrs().test(a1) && symbol.attrs().test(a2)) {
    messages_.Say("'%s' may not have both the %s and %s attributes"_err_en_US,
        symbol.name(), EnumToString(a1), EnumToString(a2));
    return true;
  } else {
    return false;
  }
}

std::optional<std::vector<Procedure>> CheckHelper::Characterize(
    const SymbolVector &specifics) {
  std::vector<Procedure> result;
  for (const Symbol &specific : specifics) {
    auto proc{Procedure::Characterize(specific, context_.intrinsics())};
    if (!proc || context_.HasError(specific)) {
      return std::nullopt;
    }
    result.emplace_back(*proc);
  }
  return result;
}

void CheckHelper::CheckVolatile(const Symbol &symbol, bool isAssociated,
    const DerivedTypeSpec *derived) { // C866 - C868
  if (IsIntentIn(symbol)) {
    messages_.Say(
        "VOLATILE attribute may not apply to an INTENT(IN) argument"_err_en_US);
  }
  if (IsProcedure(symbol)) {
    messages_.Say("VOLATILE attribute may apply only to a variable"_err_en_US);
  }
  if (isAssociated) {
    const Symbol &ultimate{symbol.GetUltimate()};
    if (IsCoarray(ultimate)) {
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
  if (const Symbol * dtSymbol{dtScope.symbol()}) {
    if (symbol.attrs().test(Attr::DEFERRED)) {
      if (!dtSymbol->attrs().test(Attr::ABSTRACT)) { // C733
        SayWithDeclaration(*dtSymbol,
            "Procedure bound to non-ABSTRACT derived type '%s' may not be DEFERRED"_err_en_US,
            dtSymbol->name());
      }
      if (symbol.attrs().test(Attr::NON_OVERRIDABLE)) {
        messages_.Say(
            "Type-bound procedure '%s' may not be both DEFERRED and NON_OVERRIDABLE"_err_en_US,
            symbol.name());
      }
    }
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
        auto bindingChars{evaluate::characteristics::Procedure::Characterize(
            binding.symbol(), context_.intrinsics())};
        auto overriddenChars{evaluate::characteristics::Procedure::Characterize(
            overriddenBinding->symbol(), context_.intrinsics())};
        if (bindingChars && overriddenChars) {
          if (isNopass) {
            if (!bindingChars->CanOverride(*overriddenChars, std::nullopt)) {
              SayWithDeclaration(*overridden,
                  "A type-bound procedure and its override must have compatible interfaces"_err_en_US);
            }
          } else {
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
  common::Restorer<const Symbol *> restorer{innermostSymbol_};
  if (const Symbol * symbol{scope.symbol()}) {
    innermostSymbol_ = symbol;
  } else if (scope.IsDerivedType()) {
    return; // PDT instantiations have null symbol()
  }
  for (const auto &set : scope.equivalenceSets()) {
    CheckEquivalenceSet(set);
  }
  for (const auto &pair : scope) {
    Check(*pair.second);
  }
  for (const Scope &child : scope.children()) {
    Check(child);
  }
  if (scope.kind() == Scope::Kind::BlockData) {
    CheckBlockData(scope);
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
  MaybeExpr bindName1{details1.bindName()};
  MaybeExpr bindName2{details2.bindName()};
  if (bindName1.has_value() != bindName2.has_value()) {
    Say(symbol1, symbol2,
        bindName1.has_value()
            ? "Module subprogram '%s' has a binding label but the corresponding"
              " interface body does not"_err_en_US
            : "Module subprogram '%s' does not have a binding label but the"
              " corresponding interface body does"_err_en_US);
  } else if (bindName1) {
    std::string string1{bindName1->AsFortran()};
    std::string string2{bindName2->AsFortran()};
    if (string1 != string2) {
      Say(symbol1, symbol2,
          "Module subprogram '%s' has binding label %s but the corresponding"
          " interface body has %s"_err_en_US,
          string1, string2);
    }
  }
  auto proc1{Procedure::Characterize(symbol1, context.intrinsics())};
  auto proc2{Procedure::Characterize(symbol2, context.intrinsics())};
  if (!proc1 || !proc2) {
    return;
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
  std::visit(common::visitors{
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
    parser::MessageFixedText &&text, A &&... args) {
  auto &message{context.Say(symbol1.name(), std::move(text), symbol1.name(),
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
  return evaluate::characteristics::ShapesAreCompatible(
      FoldShape(obj1.type.shape()), FoldShape(obj2.type.shape()));
}

evaluate::Shape SubprogramMatchHelper::FoldShape(const evaluate::Shape &shape) {
  evaluate::Shape result;
  for (const auto &extent : shape) {
    result.emplace_back(
        evaluate::Fold(context.foldingContext(), common::Clone(extent)));
  }
  return result;
}

void CheckDeclarations(SemanticsContext &context) {
  CheckHelper{context}.Check();
}

} // namespace Fortran::semantics
