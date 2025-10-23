//===-- lib/Semantics/check-call.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-call.h"
#include "definable.h"
#include "pointer-assignment.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/check-expression.h"
#include "flang/Evaluate/fold-designator.h"
#include "flang/Evaluate/shape.h"
#include "flang/Evaluate/tools.h"
#include "flang/Parser/characters.h"
#include "flang/Parser/message.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/tools.h"
#include <map>
#include <string>

using namespace Fortran::parser::literals;
namespace characteristics = Fortran::evaluate::characteristics;

namespace Fortran::semantics {

static void CheckImplicitInterfaceArg(evaluate::ActualArgument &arg,
    parser::ContextualMessages &messages, SemanticsContext &context) {
  auto restorer{
      messages.SetLocation(arg.sourceLocation().value_or(messages.at()))};
  if (auto kw{arg.keyword()}) {
    messages.Say(*kw,
        "Keyword '%s=' may not appear in a reference to a procedure with an implicit interface"_err_en_US,
        *kw);
  }
  auto type{arg.GetType()};
  if (type) {
    if (type->IsAssumedType()) {
      messages.Say(
          "Assumed type actual argument requires an explicit interface"_err_en_US);
    } else if (type->IsUnlimitedPolymorphic()) {
      messages.Say(
          "Unlimited polymorphic actual argument requires an explicit interface"_err_en_US);
    } else if (const DerivedTypeSpec * derived{GetDerivedTypeSpec(type)}) {
      if (!derived->parameters().empty()) {
        messages.Say(
            "Parameterized derived type actual argument requires an explicit interface"_err_en_US);
      }
    }
  }
  if (arg.isPercentVal() &&
      (!type || !type->IsLengthlessIntrinsicType() || arg.Rank() != 0)) {
    messages.Say(
        "%VAL argument must be a scalar numeric or logical expression"_err_en_US);
  }
  if (const auto *expr{arg.UnwrapExpr()}) {
    if (const Symbol * base{GetFirstSymbol(*expr)};
        base && IsFunctionResult(*base)) {
      context.NoteDefinedSymbol(*base);
    }
    if (IsBOZLiteral(*expr)) {
      messages.Say("BOZ argument requires an explicit interface"_err_en_US);
    } else if (evaluate::IsNullPointerOrAllocatable(expr)) {
      messages.Say(
          "Null pointer argument requires an explicit interface"_err_en_US);
    } else if (auto named{evaluate::ExtractNamedEntity(*expr)}) {
      const Symbol &symbol{named->GetLastSymbol()};
      if (IsAssumedRank(symbol)) {
        messages.Say(
            "Assumed rank argument requires an explicit interface"_err_en_US);
      }
      if (symbol.attrs().test(Attr::ASYNCHRONOUS)) {
        messages.Say(
            "ASYNCHRONOUS argument requires an explicit interface"_err_en_US);
      }
      if (symbol.attrs().test(Attr::VOLATILE)) {
        messages.Say(
            "VOLATILE argument requires an explicit interface"_err_en_US);
      }
    } else if (auto argChars{characteristics::DummyArgument::FromActual(
                   "actual argument", *expr, context.foldingContext(),
                   /*forImplicitInterface=*/true)}) {
      const auto *argProcDesignator{
          std::get_if<evaluate::ProcedureDesignator>(&expr->u)};
      if (const auto *argProcSymbol{
              argProcDesignator ? argProcDesignator->GetSymbol() : nullptr}) {
        if (!argChars->IsTypelessIntrinsicDummy() && argProcDesignator &&
            argProcDesignator->IsElemental()) { // C1533
          evaluate::SayWithDeclaration(messages, *argProcSymbol,
              "Non-intrinsic ELEMENTAL procedure '%s' may not be passed as an actual argument"_err_en_US,
              argProcSymbol->name());
        } else if (const auto *subp{argProcSymbol->GetUltimate()
                                        .detailsIf<SubprogramDetails>()}) {
          if (subp->stmtFunction()) {
            evaluate::SayWithDeclaration(messages, *argProcSymbol,
                "Statement function '%s' may not be passed as an actual argument"_err_en_US,
                argProcSymbol->name());
          }
        }
      }
    }
  }
}

// F'2023 15.5.2.12p1: "Sequence association only applies when the dummy
// argument is an explicit-shape or assumed-size array."
static bool CanAssociateWithStorageSequence(
    const characteristics::DummyDataObject &dummy) {
  return !dummy.type.attrs().test(
             characteristics::TypeAndShape::Attr::AssumedRank) &&
      !dummy.type.attrs().test(
          characteristics::TypeAndShape::Attr::AssumedShape) &&
      !dummy.attrs.test(characteristics::DummyDataObject::Attr::Allocatable) &&
      !dummy.attrs.test(characteristics::DummyDataObject::Attr::Pointer) &&
      dummy.type.corank() == 0;
}

// When a CHARACTER actual argument is known to be short,
// we extend it on the right with spaces and a warning if
// possible.  When it is long, and not required to be equal,
// the usage conforms to the standard and no warning is needed.
static void CheckCharacterActual(evaluate::Expr<evaluate::SomeType> &actual,
    const characteristics::DummyDataObject &dummy,
    characteristics::TypeAndShape &actualType, SemanticsContext &context,
    parser::ContextualMessages &messages, bool extentErrors,
    const std::string &dummyName) {
  if (dummy.type.type().category() == TypeCategory::Character &&
      actualType.type().category() == TypeCategory::Character &&
      dummy.type.type().kind() == actualType.type().kind() &&
      !dummy.attrs.test(
          characteristics::DummyDataObject::Attr::DeducedFromActual)) {
    bool actualIsAssumedRank{IsAssumedRank(actual)};
    if (actualIsAssumedRank &&
        !dummy.type.attrs().test(
            characteristics::TypeAndShape::Attr::AssumedRank)) {
      if (!context.languageFeatures().IsEnabled(
              common::LanguageFeature::AssumedRankPassedToNonAssumedRank)) {
        messages.Say(
            "Assumed-rank character array may not be associated with a dummy argument that is not assumed-rank"_err_en_US);
      } else {
        context.Warn(common::LanguageFeature::AssumedRankPassedToNonAssumedRank,
            messages.at(),
            "Assumed-rank character array should not be associated with a dummy argument that is not assumed-rank"_port_en_US);
      }
    }
    if (dummy.type.LEN() && actualType.LEN()) {
      evaluate::FoldingContext &foldingContext{context.foldingContext()};
      auto dummyLength{
          ToInt64(Fold(foldingContext, common::Clone(*dummy.type.LEN())))};
      auto actualLength{
          ToInt64(Fold(foldingContext, common::Clone(*actualType.LEN())))};
      if (dummyLength && actualLength) {
        bool canAssociate{CanAssociateWithStorageSequence(dummy)};
        if (dummy.type.Rank() > 0 && canAssociate) {
          // Character storage sequence association (F'2023 15.5.2.12p4)
          if (auto dummySize{evaluate::ToInt64(evaluate::Fold(
                  foldingContext, evaluate::GetSize(dummy.type.shape())))}) {
            auto dummyChars{*dummySize * *dummyLength};
            if (actualType.Rank() == 0 && !actualIsAssumedRank) {
              evaluate::DesignatorFolder folder{
                  context.foldingContext(), /*getLastComponent=*/true};
              if (auto actualOffset{folder.FoldDesignator(actual)}) {
                std::int64_t actualChars{*actualLength};
                if (IsAllocatableOrPointer(actualOffset->symbol())) {
                  // don't use actualOffset->symbol().size()!
                } else if (static_cast<std::size_t>(actualOffset->offset()) >=
                        actualOffset->symbol().size() ||
                    !evaluate::IsContiguous(
                        actualOffset->symbol(), foldingContext)) {
                  // If substring, take rest of substring
                  if (*actualLength > 0) {
                    actualChars -=
                        (actualOffset->offset() / actualType.type().kind()) %
                        *actualLength;
                  }
                } else {
                  actualChars = (static_cast<std::int64_t>(
                                     actualOffset->symbol().size()) -
                                    actualOffset->offset()) /
                      actualType.type().kind();
                }
                if (actualChars < dummyChars) {
                  if (extentErrors) {
                    messages.Say(
                        "Actual argument has fewer characters remaining in storage sequence (%jd) than %s (%jd)"_err_en_US,
                        static_cast<std::intmax_t>(actualChars), dummyName,
                        static_cast<std::intmax_t>(dummyChars));
                  } else if (context.ShouldWarn(
                                 common::UsageWarning::ShortCharacterActual)) {
                    messages.Say(common::UsageWarning::ShortCharacterActual,
                        "Actual argument has fewer characters remaining in storage sequence (%jd) than %s (%jd)"_warn_en_US,
                        static_cast<std::intmax_t>(actualChars), dummyName,
                        static_cast<std::intmax_t>(dummyChars));
                  }
                }
              }
            } else { // actual.type.Rank() > 0
              if (auto actualSize{evaluate::ToInt64(evaluate::Fold(
                      foldingContext, evaluate::GetSize(actualType.shape())))};
                  actualSize &&
                  *actualSize * *actualLength < *dummySize * *dummyLength) {
                if (extentErrors) {
                  messages.Say(
                      "Actual argument array has fewer characters (%jd) than %s array (%jd)"_err_en_US,
                      static_cast<std::intmax_t>(*actualSize * *actualLength),
                      dummyName,
                      static_cast<std::intmax_t>(*dummySize * *dummyLength));
                } else if (context.ShouldWarn(
                               common::UsageWarning::ShortCharacterActual)) {
                  messages.Say(common::UsageWarning::ShortCharacterActual,
                      "Actual argument array has fewer characters (%jd) than %s array (%jd)"_warn_en_US,
                      static_cast<std::intmax_t>(*actualSize * *actualLength),
                      dummyName,
                      static_cast<std::intmax_t>(*dummySize * *dummyLength));
                }
              }
            }
          }
        } else if (*actualLength != *dummyLength) {
          // Not using storage sequence association, and the lengths don't
          // match.
          if (!canAssociate) {
            // F'2023 15.5.2.5 paragraph 4
            messages.Say(
                "Actual argument variable length '%jd' does not match the expected length '%jd'"_err_en_US,
                *actualLength, *dummyLength);
          } else if (*actualLength < *dummyLength) {
            CHECK(dummy.type.Rank() == 0);
            bool isVariable{evaluate::IsVariable(actual)};
            if (context.ShouldWarn(
                    common::UsageWarning::ShortCharacterActual)) {
              if (isVariable) {
                messages.Say(common::UsageWarning::ShortCharacterActual,
                    "Actual argument variable length '%jd' is less than expected length '%jd'"_warn_en_US,
                    *actualLength, *dummyLength);
              } else {
                messages.Say(common::UsageWarning::ShortCharacterActual,
                    "Actual argument expression length '%jd' is less than expected length '%jd'"_warn_en_US,
                    *actualLength, *dummyLength);
              }
            }
            if (!isVariable) {
              auto converted{
                  ConvertToType(dummy.type.type(), std::move(actual))};
              CHECK(converted);
              actual = std::move(*converted);
              actualType.set_LEN(SubscriptIntExpr{*dummyLength});
            }
          }
        }
      }
    }
  }
}

// Automatic conversion of different-kind INTEGER scalar actual
// argument expressions (not variables) to INTEGER scalar dummies.
// We return nonstandard INTEGER(8) results from intrinsic functions
// like SIZE() by default in order to facilitate the use of large
// arrays.  Emit a warning when downconverting.
static void ConvertIntegerActual(evaluate::Expr<evaluate::SomeType> &actual,
    const characteristics::TypeAndShape &dummyType,
    characteristics::TypeAndShape &actualType,
    parser::ContextualMessages &messages, SemanticsContext &semanticsContext) {
  if (dummyType.type().category() == TypeCategory::Integer &&
      actualType.type().category() == TypeCategory::Integer &&
      dummyType.type().kind() != actualType.type().kind() &&
      dummyType.Rank() == 0 && actualType.Rank() == 0 &&
      !evaluate::IsVariable(actual)) {
    auto converted{
        evaluate::ConvertToType(dummyType.type(), std::move(actual))};
    CHECK(converted);
    actual = std::move(*converted);
    if (dummyType.type().kind() < actualType.type().kind()) {
      if (!semanticsContext.IsEnabled(
              common::LanguageFeature::ActualIntegerConvertedToSmallerKind)) {
        messages.Say(
            "Actual argument scalar expression of type INTEGER(%d) cannot be implicitly converted to smaller dummy argument type INTEGER(%d)"_err_en_US,
            actualType.type().kind(), dummyType.type().kind());
      } else if (semanticsContext.ShouldWarn(common::LanguageFeature::
                         ActualIntegerConvertedToSmallerKind)) {
        messages.Say(
            common::LanguageFeature::ActualIntegerConvertedToSmallerKind,
            "Actual argument scalar expression of type INTEGER(%d) was converted to smaller dummy argument type INTEGER(%d)"_port_en_US,
            actualType.type().kind(), dummyType.type().kind());
      }
    }
    actualType = dummyType;
  }
}

// Automatic conversion of different-kind LOGICAL scalar actual argument
// expressions (not variables) to LOGICAL scalar dummies when the dummy is of
// default logical kind. This allows expressions in dummy arguments to work when
// the default logical kind is not the one used in LogicalResult. This will
// always be safe even when downconverting so no warning is needed.
static void ConvertLogicalActual(evaluate::Expr<evaluate::SomeType> &actual,
    const characteristics::TypeAndShape &dummyType,
    characteristics::TypeAndShape &actualType) {
  if (dummyType.type().category() == TypeCategory::Logical &&
      actualType.type().category() == TypeCategory::Logical &&
      dummyType.type().kind() != actualType.type().kind() &&
      !evaluate::IsVariable(actual)) {
    auto converted{
        evaluate::ConvertToType(dummyType.type(), std::move(actual))};
    CHECK(converted);
    actual = std::move(*converted);
    actualType = dummyType;
  }
}

static bool DefersSameTypeParameters(
    const DerivedTypeSpec *actual, const DerivedTypeSpec *dummy) {
  if (actual && dummy) {
    for (const auto &pair : actual->parameters()) {
      const ParamValue &actualValue{pair.second};
      const ParamValue *dummyValue{dummy->FindParameter(pair.first)};
      if (!dummyValue ||
          (actualValue.isDeferred() != dummyValue->isDeferred())) {
        return false;
      }
    }
  }
  return true;
}

static void CheckExplicitDataArg(const characteristics::DummyDataObject &dummy,
    const std::string &dummyName, evaluate::Expr<evaluate::SomeType> &actual,
    characteristics::TypeAndShape &actualType, bool isElemental,
    SemanticsContext &context, evaluate::FoldingContext &foldingContext,
    const Scope *scope, const evaluate::SpecificIntrinsic *intrinsic,
    bool allowActualArgumentConversions, bool extentErrors,
    const characteristics::Procedure &procedure,
    const evaluate::ActualArgument &arg) {

  // Basic type & rank checking
  parser::ContextualMessages &messages{foldingContext.messages()};
  CheckCharacterActual(
      actual, dummy, actualType, context, messages, extentErrors, dummyName);
  bool dummyIsAllocatable{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Allocatable)};
  bool dummyIsPointer{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Pointer)};
  bool dummyIsAllocatableOrPointer{dummyIsAllocatable || dummyIsPointer};
  allowActualArgumentConversions &= !dummyIsAllocatableOrPointer;
  bool typesCompatibleWithIgnoreTKR{
      (dummy.ignoreTKR.test(common::IgnoreTKR::Type) &&
          (dummy.type.type().category() == TypeCategory::Derived ||
              actualType.type().category() == TypeCategory::Derived ||
              dummy.type.type().category() != actualType.type().category())) ||
      (dummy.ignoreTKR.test(common::IgnoreTKR::Kind) &&
          dummy.type.type().category() == actualType.type().category())};
  allowActualArgumentConversions &= !typesCompatibleWithIgnoreTKR;
  if (allowActualArgumentConversions) {
    ConvertIntegerActual(actual, dummy.type, actualType, messages, context);
    ConvertLogicalActual(actual, dummy.type, actualType);
  }
  bool typesCompatible{typesCompatibleWithIgnoreTKR ||
      dummy.type.type().IsTkCompatibleWith(actualType.type())};
  int dummyRank{dummy.type.Rank()};
  if (typesCompatible) {
    if (const auto *constantChar{
            evaluate::UnwrapConstantValue<evaluate::Ascii>(actual)};
        constantChar && constantChar->wasHollerith() &&
        dummy.type.type().IsUnlimitedPolymorphic() &&
        context.ShouldWarn(common::LanguageFeature::HollerithPolymorphic)) {
      messages.Say(common::LanguageFeature::HollerithPolymorphic,
          "passing Hollerith to unlimited polymorphic as if it were CHARACTER"_port_en_US);
    }
  } else if (dummyRank == 0 && allowActualArgumentConversions) {
    // Extension: pass Hollerith literal to scalar as if it had been BOZ
    if (auto converted{evaluate::HollerithToBOZ(
            foldingContext, actual, dummy.type.type())}) {
      if (context.ShouldWarn(
              common::LanguageFeature::HollerithOrCharacterAsBOZ)) {
        messages.Say(common::LanguageFeature::HollerithOrCharacterAsBOZ,
            "passing Hollerith or character literal as if it were BOZ"_port_en_US);
      }
      actual = *converted;
      actualType.type() = dummy.type.type();
      typesCompatible = true;
    }
  }
  bool dummyIsAssumedRank{dummy.type.attrs().test(
      characteristics::TypeAndShape::Attr::AssumedRank)};
  bool actualIsAssumedSize{actualType.attrs().test(
      characteristics::TypeAndShape::Attr::AssumedSize)};
  bool actualIsAssumedRank{IsAssumedRank(actual)};
  bool actualIsPointer{evaluate::IsObjectPointer(actual)};
  bool actualIsAllocatable{evaluate::IsAllocatableDesignator(actual)};
  bool actualMayBeAssumedSize{actualIsAssumedSize ||
      (actualIsAssumedRank && !actualIsPointer && !actualIsAllocatable)};
  bool actualIsPolymorphic{actualType.type().IsPolymorphic()};
  const auto *actualDerived{evaluate::GetDerivedTypeSpec(actualType.type())};
  if (typesCompatible) {
    if (isElemental) {
    } else if (dummyIsAssumedRank) {
      if (actualMayBeAssumedSize && dummy.intent == common::Intent::Out) {
        // An INTENT(OUT) dummy might be a no-op at run time
        bool dummyHasSignificantIntentOut{actualIsPolymorphic ||
            (actualDerived &&
                (actualDerived->HasDefaultInitialization(
                     /*ignoreAllocatable=*/false, /*ignorePointer=*/true) ||
                    actualDerived->HasDestruction()))};
        const char *actualDesc{
            actualIsAssumedSize ? "Assumed-size" : "Assumed-rank"};
        if (dummyHasSignificantIntentOut) {
          messages.Say(
              "%s actual argument may not be associated with INTENT(OUT) assumed-rank dummy argument requiring finalization, destruction, or initialization"_err_en_US,
              actualDesc);
        } else {
          context.Warn(common::UsageWarning::Portability, messages.at(),
              "%s actual argument should not be associated with INTENT(OUT) assumed-rank dummy argument"_port_en_US,
              actualDesc);
        }
      }
    } else if (dummy.ignoreTKR.test(common::IgnoreTKR::Rank)) {
    } else if (dummyRank > 0 && !dummyIsAllocatableOrPointer &&
        !dummy.type.attrs().test(
            characteristics::TypeAndShape::Attr::AssumedShape) &&
        !dummy.type.attrs().test(
            characteristics::TypeAndShape::Attr::DeferredShape) &&
        (actualType.Rank() > 0 || IsArrayElement(actual))) {
      // Sequence association (15.5.2.11) applies -- rank need not match
      // if the actual argument is an array or array element designator,
      // and the dummy is an array, but not assumed-shape or an INTENT(IN)
      // pointer that's standing in for an assumed-shape dummy.
    } else if (dummy.type.shape() && actualType.shape()) {
      // Let CheckConformance accept actual scalars; storage association
      // cases are checked here below.
      CheckConformance(messages, *dummy.type.shape(), *actualType.shape(),
          dummyIsAllocatableOrPointer
              ? evaluate::CheckConformanceFlags::None
              : evaluate::CheckConformanceFlags::RightScalarExpandable,
          "dummy argument", "actual argument");
    }
  } else {
    const auto &len{actualType.LEN()};
    messages.Say(
        "Actual argument type '%s' is not compatible with dummy argument type '%s'"_err_en_US,
        actualType.type().AsFortran(len ? len->AsFortran() : ""),
        dummy.type.type().AsFortran());
  }

  auto actualCoarrayRef{ExtractCoarrayRef(actual)};
  bool dummyIsAssumedSize{dummy.type.attrs().test(
      characteristics::TypeAndShape::Attr::AssumedSize)};
  bool dummyIsAsynchronous{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Asynchronous)};
  bool dummyIsVolatile{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Volatile)};
  bool dummyIsValue{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Value)};
  bool dummyIsPolymorphic{dummy.type.type().IsPolymorphic()};
  if (actualIsPolymorphic && dummyIsPolymorphic &&
      actualCoarrayRef) { // 15.5.2.4(2)
    messages.Say(
        "Coindexed polymorphic object may not be associated with a polymorphic %s"_err_en_US,
        dummyName);
  }
  if (actualIsPolymorphic && !dummyIsPolymorphic &&
      actualIsAssumedSize) { // 15.5.2.4(2)
    messages.Say(
        "Assumed-size polymorphic array may not be associated with a monomorphic %s"_err_en_US,
        dummyName);
  }

  // Derived type actual argument checks
  const Symbol *actualFirstSymbol{evaluate::GetFirstSymbol(actual)};
  bool actualIsAsynchronous{
      actualFirstSymbol && actualFirstSymbol->attrs().test(Attr::ASYNCHRONOUS)};
  bool actualIsVolatile{
      actualFirstSymbol && actualFirstSymbol->attrs().test(Attr::VOLATILE)};
  if (actualDerived && !actualDerived->IsVectorType()) {
    if (dummy.type.type().IsAssumedType()) {
      if (!actualDerived->parameters().empty()) { // 15.5.2.4(2)
        messages.Say(
            "Actual argument associated with TYPE(*) %s may not have a parameterized derived type"_err_en_US,
            dummyName);
      }
      if (const Symbol *
          tbp{FindImmediateComponent(*actualDerived, [](const Symbol &symbol) {
            return symbol.has<ProcBindingDetails>();
          })}) { // 15.5.2.4(2)
        evaluate::SayWithDeclaration(messages, *tbp,
            "Actual argument associated with TYPE(*) %s may not have type-bound procedure '%s'"_err_en_US,
            dummyName, tbp->name());
      }
      auto finals{FinalsForDerivedTypeInstantiation(*actualDerived)};
      if (!finals.empty()) { // 15.5.2.4(2)
        SourceName name{finals.front()->name()};
        if (auto *msg{messages.Say(
                "Actual argument associated with TYPE(*) %s may not have derived type '%s' with FINAL subroutine '%s'"_err_en_US,
                dummyName, actualDerived->typeSymbol().name(), name)}) {
          msg->Attach(name, "FINAL subroutine '%s' in derived type '%s'"_en_US,
              name, actualDerived->typeSymbol().name());
        }
      }
    }
    if (actualCoarrayRef) {
      if (dummy.intent != common::Intent::In && !dummyIsValue) {
        if (auto bad{FindAllocatableUltimateComponent(
                *actualDerived)}) { // 15.5.2.4(6)
          evaluate::SayWithDeclaration(messages, *bad,
              "Coindexed actual argument with ALLOCATABLE ultimate component '%s' must be associated with a %s with VALUE or INTENT(IN) attributes"_err_en_US,
              bad.BuildResultDesignatorName(), dummyName);
        }
      }
      const Symbol &coarray{actualCoarrayRef->GetLastSymbol()};
      if (const DeclTypeSpec * type{coarray.GetType()}) { // C1537
        if (const DerivedTypeSpec * derived{type->AsDerived()}) {
          if (auto bad{semantics::FindPointerUltimateComponent(*derived)}) {
            evaluate::SayWithDeclaration(messages, coarray,
                "Coindexed object '%s' with POINTER ultimate component '%s' cannot be associated with %s"_err_en_US,
                coarray.name(), bad.BuildResultDesignatorName(), dummyName);
          }
        }
      }
    }
    if (actualIsVolatile != dummyIsVolatile) { // 15.5.2.4(22)
      if (auto bad{semantics::FindCoarrayUltimateComponent(*actualDerived)}) {
        evaluate::SayWithDeclaration(messages, *bad,
            "VOLATILE attribute must match for %s when actual argument has a coarray ultimate component '%s'"_err_en_US,
            dummyName, bad.BuildResultDesignatorName());
      }
    }
  }

  // Rank and shape checks
  const auto *actualLastSymbol{evaluate::GetLastSymbol(actual)};
  if (actualLastSymbol) {
    actualLastSymbol = &ResolveAssociations(*actualLastSymbol);
  }
  int actualRank{actualType.Rank()};
  if (dummy.type.attrs().test(
          characteristics::TypeAndShape::Attr::AssumedShape)) {
    // 15.5.2.4(16)
    if (actualIsAssumedRank) {
      messages.Say(
          "Assumed-rank actual argument may not be associated with assumed-shape %s"_err_en_US,
          dummyName);
    } else if (actualRank == 0) {
      messages.Say(
          "Scalar actual argument may not be associated with assumed-shape %s"_err_en_US,
          dummyName);
    } else if (actualIsAssumedSize && actualLastSymbol) {
      evaluate::SayWithDeclaration(messages, *actualLastSymbol,
          "Assumed-size array may not be associated with assumed-shape %s"_err_en_US,
          dummyName);
    }
  } else if (dummyRank > 0) {
    bool basicError{false};
    if (actualRank == 0 && !actualIsAssumedRank &&
        !dummyIsAllocatableOrPointer) {
      // Actual is scalar, dummy is an array.  F'2023 15.5.2.5p14
      if (actualCoarrayRef) {
        basicError = true;
        messages.Say(
            "Coindexed scalar actual argument must be associated with a scalar %s"_err_en_US,
            dummyName);
      }
      bool actualIsArrayElement{IsArrayElement(actual) != nullptr};
      bool actualIsCKindCharacter{
          actualType.type().category() == TypeCategory::Character &&
          actualType.type().kind() == 1};
      if (!actualIsCKindCharacter) {
        if (!actualIsArrayElement &&
            !(dummy.type.type().IsAssumedType() && dummyIsAssumedSize) &&
            !dummyIsAssumedRank &&
            !dummy.ignoreTKR.test(common::IgnoreTKR::Rank)) {
          basicError = true;
          messages.Say(
              "Whole scalar actual argument may not be associated with a %s array"_err_en_US,
              dummyName);
        }
        if (actualIsPolymorphic) {
          basicError = true;
          messages.Say(
              "Polymorphic scalar may not be associated with a %s array"_err_en_US,
              dummyName);
        }
        bool isOkBecauseContiguous{
            context.IsEnabled(
                common::LanguageFeature::ContiguousOkForSeqAssociation) &&
            actualLastSymbol &&
            evaluate::IsContiguous(*actualLastSymbol, foldingContext)};
        if (actualIsArrayElement && actualLastSymbol &&
            !dummy.ignoreTKR.test(common::IgnoreTKR::Contiguous)) {
          if (IsPointer(*actualLastSymbol)) {
            if (isOkBecauseContiguous) {
              context.Warn(
                  common::LanguageFeature::ContiguousOkForSeqAssociation,
                  messages.at(),
                  "Element of contiguous pointer array is accepted for storage sequence association"_port_en_US);
            } else {
              basicError = true;
              messages.Say(
                  "Element of pointer array may not be associated with a %s array"_err_en_US,
                  dummyName);
            }
          } else if (IsAssumedShape(*actualLastSymbol) &&
              !dummy.ignoreTKR.test(common::IgnoreTKR::Contiguous)) {
            if (isOkBecauseContiguous) {
              context.Warn(
                  common::LanguageFeature::ContiguousOkForSeqAssociation,
                  messages.at(),
                  "Element of contiguous assumed-shape array is accepted for storage sequence association"_port_en_US);
            } else {
              basicError = true;
              messages.Say(
                  "Element of assumed-shape array may not be associated with a %s array"_err_en_US,
                  dummyName);
            }
          }
        }
      }
    }
    // Storage sequence association (F'2023 15.5.2.12p3) checks.
    // Character storage sequence association is checked in
    // CheckCharacterActual().
    if (!basicError &&
        actualType.type().category() != TypeCategory::Character &&
        CanAssociateWithStorageSequence(dummy) &&
        !dummy.attrs.test(
            characteristics::DummyDataObject::Attr::DeducedFromActual)) {
      if (auto dummySize{evaluate::ToInt64(evaluate::Fold(
              foldingContext, evaluate::GetSize(dummy.type.shape())))}) {
        if (actualIsAssumedRank) {
          if (!context.languageFeatures().IsEnabled(
                  common::LanguageFeature::AssumedRankPassedToNonAssumedRank)) {
            messages.Say(
                "Assumed-rank array may not be associated with a dummy argument that is not assumed-rank"_err_en_US);
          } else {
            context.Warn(
                common::LanguageFeature::AssumedRankPassedToNonAssumedRank,
                messages.at(),
                "Assumed-rank array should not be associated with a dummy argument that is not assumed-rank"_port_en_US);
          }
        } else if (actualRank == 0) {
          if (evaluate::IsArrayElement(actual)) {
            // Actual argument is a scalar array element
            evaluate::DesignatorFolder folder{
                context.foldingContext(), /*getLastComponent=*/true};
            if (auto actualOffset{folder.FoldDesignator(actual)}) {
              std::optional<std::int64_t> actualElements;
              if (IsAllocatableOrPointer(actualOffset->symbol())) {
                // don't use actualOffset->symbol().size()!
              } else if (static_cast<std::size_t>(actualOffset->offset()) >=
                      actualOffset->symbol().size() ||
                  !evaluate::IsContiguous(
                      actualOffset->symbol(), foldingContext)) {
                actualElements = 1;
              } else if (auto actualSymType{evaluate::DynamicType::From(
                             actualOffset->symbol())}) {
                if (auto actualSymTypeBytes{
                        evaluate::ToInt64(evaluate::Fold(foldingContext,
                            actualSymType->MeasureSizeInBytes(
                                foldingContext, false)))};
                    actualSymTypeBytes && *actualSymTypeBytes > 0) {
                  actualElements = (static_cast<std::int64_t>(
                                        actualOffset->symbol().size()) -
                                       actualOffset->offset()) /
                      *actualSymTypeBytes;
                }
              }
              if (actualElements && *actualElements < *dummySize) {
                if (extentErrors) {
                  messages.Say(
                      "Actual argument has fewer elements remaining in storage sequence (%jd) than %s array (%jd)"_err_en_US,
                      static_cast<std::intmax_t>(*actualElements), dummyName,
                      static_cast<std::intmax_t>(*dummySize));
                } else if (context.ShouldWarn(
                               common::UsageWarning::ShortArrayActual)) {
                  messages.Say(common::UsageWarning::ShortArrayActual,
                      "Actual argument has fewer elements remaining in storage sequence (%jd) than %s array (%jd)"_warn_en_US,
                      static_cast<std::intmax_t>(*actualElements), dummyName,
                      static_cast<std::intmax_t>(*dummySize));
                }
              }
            }
          }
        } else {
          if (auto actualSize{evaluate::ToInt64(evaluate::Fold(
                  foldingContext, evaluate::GetSize(actualType.shape())))};
              actualSize && *actualSize < *dummySize) {
            if (extentErrors) {
              messages.Say(
                  "Actual argument array has fewer elements (%jd) than %s array (%jd)"_err_en_US,
                  static_cast<std::intmax_t>(*actualSize), dummyName,
                  static_cast<std::intmax_t>(*dummySize));
            } else if (context.ShouldWarn(
                           common::UsageWarning::ShortArrayActual)) {
              messages.Say(common::UsageWarning::ShortArrayActual,
                  "Actual argument array has fewer elements (%jd) than %s array (%jd)"_warn_en_US,
                  static_cast<std::intmax_t>(*actualSize), dummyName,
                  static_cast<std::intmax_t>(*dummySize));
            }
          }
        }
      }
    }
  }
  const ObjectEntityDetails *actualLastObject{actualLastSymbol
          ? actualLastSymbol->detailsIf<ObjectEntityDetails>()
          : nullptr};
  if (actualLastObject && actualLastObject->IsCoarray() &&
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Allocatable) &&
      dummy.intent == common::Intent::Out &&
      !(intrinsic &&
          evaluate::AcceptsIntentOutAllocatableCoarray(
              intrinsic->name))) { // C846
    messages.Say(
        "ALLOCATABLE coarray '%s' may not be associated with INTENT(OUT) %s"_err_en_US,
        actualLastSymbol->name(), dummyName);
  }

  // Definability checking
  // Problems with polymorphism are caught in the callee's definition.
  if (scope) {
    std::optional<parser::MessageFixedText> undefinableMessage;
    DefinabilityFlags flags{DefinabilityFlag::PolymorphicOkInPure};
    if (dummy.intent == common::Intent::InOut) {
      flags.set(DefinabilityFlag::AllowEventLockOrNotifyType);
      undefinableMessage =
          "Actual argument associated with INTENT(IN OUT) %s is not definable"_err_en_US;
    } else if (dummy.intent == common::Intent::Out) {
      undefinableMessage =
          "Actual argument associated with INTENT(OUT) %s is not definable"_err_en_US;
    } else if (context.ShouldWarn(common::LanguageFeature::
                       UndefinableAsynchronousOrVolatileActual)) {
      if (dummy.attrs.test(
              characteristics::DummyDataObject::Attr::Asynchronous)) {
        undefinableMessage =
            "Actual argument associated with ASYNCHRONOUS %s is not definable"_warn_en_US;
      } else if (dummy.attrs.test(
                     characteristics::DummyDataObject::Attr::Volatile)) {
        undefinableMessage =
            "Actual argument associated with VOLATILE %s is not definable"_warn_en_US;
      }
    }
    if (undefinableMessage) {
      if (isElemental) { // 15.5.2.4(21)
        flags.set(DefinabilityFlag::VectorSubscriptIsOk);
      }
      if (actualIsPointer && dummyIsPointer) { // 19.6.8
        flags.set(DefinabilityFlag::PointerDefinition);
      }
      if (auto whyNot{WhyNotDefinable(messages.at(), *scope, flags, actual)}) {
        if (whyNot->IsFatal()) {
          if (auto *msg{messages.Say(*undefinableMessage, dummyName)}) {
            if (!msg->IsFatal()) {
              msg->set_languageFeature(common::LanguageFeature::
                      UndefinableAsynchronousOrVolatileActual);
            }
            msg->Attach(
                std::move(whyNot->set_severity(parser::Severity::Because)));
          }
        } else {
          messages.Say(std::move(*whyNot));
        }
      }
    } else if (dummy.intent != common::Intent::In ||
        (dummyIsPointer && !actualIsPointer)) {
      if (auto named{evaluate::ExtractNamedEntity(actual)}) {
        if (const Symbol & base{named->GetFirstSymbol()};
            IsFunctionResult(base)) {
          context.NoteDefinedSymbol(base);
        }
      }
    }
  }

  bool dummyIsContiguous{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Contiguous)};
  bool actualIsContiguous{IsSimplyContiguous(actual, foldingContext)};

  // Cases when temporaries might be needed but must not be permitted.
  bool dummyIsAssumedShape{dummy.type.attrs().test(
      characteristics::TypeAndShape::Attr::AssumedShape)};
  if ((actualIsAsynchronous || actualIsVolatile) &&
      (dummyIsAsynchronous || dummyIsVolatile) && !dummyIsValue) {
    if (actualCoarrayRef) { // C1538
      messages.Say(
          "Coindexed ASYNCHRONOUS or VOLATILE actual argument may not be associated with %s with ASYNCHRONOUS or VOLATILE attributes unless VALUE"_err_en_US,
          dummyName);
    }
    if ((actualRank > 0 || actualIsAssumedRank) && !actualIsContiguous) {
      if (dummyIsContiguous ||
          !(dummyIsAssumedShape || dummyIsAssumedRank ||
              (actualIsPointer && dummyIsPointer))) { // C1539 & C1540
        messages.Say(
            "ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous ASYNCHRONOUS or VOLATILE %s"_err_en_US,
            dummyName);
      }
    }
  }

  // 15.5.2.6 -- dummy is ALLOCATABLE
  bool dummyIsOptional{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Optional)};
  if (dummyIsAllocatable) {
    if (actualIsAllocatable) {
      if (actualCoarrayRef && dummy.intent != common::Intent::In) {
        messages.Say(
            "ALLOCATABLE %s must have INTENT(IN) to be associated with a coindexed actual argument"_err_en_US,
            dummyName);
      }
      if (!actualCoarrayRef && actualLastSymbol && dummy.type.corank() == 0 &&
          actualLastSymbol->Corank() > 0) {
        messages.Say(
            "ALLOCATABLE %s is not a coarray but actual argument has corank %d"_err_en_US,
            dummyName, actualLastSymbol->Corank());
      }
    } else if (evaluate::IsBareNullPointer(&actual)) {
      if (dummyIsOptional) {
      } else if (dummy.intent == common::Intent::Default &&
          context.ShouldWarn(
              common::UsageWarning::NullActualForDefaultIntentAllocatable)) {
        messages.Say(
            "A null pointer should not be associated with allocatable %s without INTENT(IN)"_warn_en_US,
            dummyName);
      } else if (dummy.intent == common::Intent::In &&
          context.ShouldWarn(
              common::LanguageFeature::NullActualForAllocatable)) {
        messages.Say(common::LanguageFeature::NullActualForAllocatable,
            "Allocatable %s is associated with a null pointer"_port_en_US,
            dummyName);
      }
      // INTENT(OUT) and INTENT(IN OUT) cases are caught elsewhere as being
      // undefinable actual arguments.
    } else if (evaluate::IsNullAllocatable(&actual)) {
      if (dummyIsOptional) {
      } else if (dummy.intent == common::Intent::Default &&
          context.ShouldWarn(
              common::UsageWarning::NullActualForDefaultIntentAllocatable)) {
        messages.Say(
            "A null allocatable should not be associated with allocatable %s without INTENT(IN)"_warn_en_US,
            dummyName);
      }
      // INTENT(OUT) and INTENT(IN OUT) cases are caught elsewhere
    } else {
      messages.Say(
          "ALLOCATABLE %s must be associated with an ALLOCATABLE actual argument"_err_en_US,
          dummyName);
    }
  }

  // 15.5.2.7 -- dummy is POINTER
  if (dummyIsPointer) {
    if (actualIsPointer || dummy.intent == common::Intent::In) {
      if (scope) {
        semantics::CheckPointerAssignment(context, messages.at(), dummyName,
            dummy, actual, *scope,
            /*isAssumedRank=*/dummyIsAssumedRank, actualIsPointer);
      }
    } else if (!actualIsPointer) {
      messages.Say(
          "Actual argument associated with POINTER %s must also be POINTER unless INTENT(IN)"_err_en_US,
          dummyName);
    }
  }

  // 15.5.2.5 -- actual & dummy are both POINTER or both ALLOCATABLE
  // For INTENT(IN), and for a polymorphic actual being associated with a
  // monomorphic dummy, we relax two checks that are in Fortran to
  // prevent the callee from changing the type or to avoid having
  // to use a descriptor.
  if (!typesCompatible) {
    // Don't pile on the errors emitted above
  } else if ((actualIsPointer && dummyIsPointer) ||
      (actualIsAllocatable && dummyIsAllocatable)) {
    bool actualIsUnlimited{actualType.type().IsUnlimitedPolymorphic()};
    bool dummyIsUnlimited{dummy.type.type().IsUnlimitedPolymorphic()};
    bool checkTypeCompatibility{true};
    if (actualIsUnlimited != dummyIsUnlimited) {
      checkTypeCompatibility = false;
      if (dummyIsUnlimited && dummy.intent == common::Intent::In &&
          context.IsEnabled(common::LanguageFeature::RelaxedIntentInChecking)) {
        if (context.ShouldWarn(
                common::LanguageFeature::RelaxedIntentInChecking)) {
          messages.Say(common::LanguageFeature::RelaxedIntentInChecking,
              "If a POINTER or ALLOCATABLE dummy or actual argument is unlimited polymorphic, both should be so"_port_en_US);
        }
      } else {
        messages.Say(
            "If a POINTER or ALLOCATABLE dummy or actual argument is unlimited polymorphic, both must be so"_err_en_US);
      }
    } else if (dummyIsPolymorphic != actualIsPolymorphic) {
      if (dummyIsPolymorphic && dummy.intent == common::Intent::In &&
          context.IsEnabled(common::LanguageFeature::RelaxedIntentInChecking)) {
        if (context.ShouldWarn(
                common::LanguageFeature::RelaxedIntentInChecking)) {
          messages.Say(common::LanguageFeature::RelaxedIntentInChecking,
              "If a POINTER or ALLOCATABLE dummy or actual argument is polymorphic, both should be so"_port_en_US);
        }
      } else if (actualIsPolymorphic &&
          context.IsEnabled(common::LanguageFeature::
                  PolymorphicActualAllocatableOrPointerToMonomorphicDummy)) {
        if (context.ShouldWarn(common::LanguageFeature::
                    PolymorphicActualAllocatableOrPointerToMonomorphicDummy)) {
          messages.Say(
              common::LanguageFeature::
                  PolymorphicActualAllocatableOrPointerToMonomorphicDummy,
              "If a POINTER or ALLOCATABLE actual argument is polymorphic, the corresponding dummy argument should also be so"_port_en_US);
        }
      } else {
        checkTypeCompatibility = false;
        messages.Say(
            "If a POINTER or ALLOCATABLE dummy or actual argument is polymorphic, both must be so"_err_en_US);
      }
    }
    if (checkTypeCompatibility && !actualIsUnlimited) {
      if (!actualType.type().IsTkCompatibleWith(dummy.type.type())) {
        if (dummy.intent == common::Intent::In &&
            context.IsEnabled(
                common::LanguageFeature::RelaxedIntentInChecking)) {
          if (context.ShouldWarn(
                  common::LanguageFeature::RelaxedIntentInChecking)) {
            messages.Say(common::LanguageFeature::RelaxedIntentInChecking,
                "POINTER or ALLOCATABLE dummy and actual arguments should have the same declared type and kind"_port_en_US);
          }
        } else {
          messages.Say(
              "POINTER or ALLOCATABLE dummy and actual arguments must have the same declared type and kind"_err_en_US);
        }
      }
      // 15.5.2.5(4)
      const auto *dummyDerived{evaluate::GetDerivedTypeSpec(dummy.type.type())};
      if (!DefersSameTypeParameters(actualDerived, dummyDerived) ||
          dummy.type.type().HasDeferredTypeParameter() !=
              actualType.type().HasDeferredTypeParameter()) {
        messages.Say(
            "Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE"_err_en_US);
      }
    }
  }

  // 15.5.2.8 -- coarray dummy arguments
  if (dummy.type.corank() > 0) {
    if (actualType.corank() == 0) {
      messages.Say(
          "Actual argument associated with coarray %s must be a coarray"_err_en_US,
          dummyName);
    } else if (actualType.corank() != dummy.type.corank() &&
        dummyIsAllocatableOrPointer) {
      messages.Say(
          "ALLOCATABLE or POINTER %s has corank %d but actual argument has corank %d"_err_en_US,
          dummyName, dummy.type.corank(), actualType.corank());
    }
    if (dummyIsVolatile) {
      if (!actualIsVolatile) {
        messages.Say(
            "non-VOLATILE coarray may not be associated with VOLATILE coarray %s"_err_en_US,
            dummyName);
      }
    } else {
      if (actualIsVolatile) {
        messages.Say(
            "VOLATILE coarray may not be associated with non-VOLATILE coarray %s"_err_en_US,
            dummyName);
      }
    }
    if (actualRank == dummyRank && !actualIsContiguous) {
      if (dummyIsContiguous) {
        messages.Say(
            "Actual argument associated with a CONTIGUOUS coarray %s must be simply contiguous"_err_en_US,
            dummyName);
      } else if (!dummyIsAssumedShape && !dummyIsAssumedRank) {
        messages.Say(
            "Actual argument associated with coarray %s (not assumed shape or rank) must be simply contiguous"_err_en_US,
            dummyName);
      }
    }
  }

  // NULL(MOLD=) checking for non-intrinsic procedures
  if (!intrinsic && !dummyIsAllocatableOrPointer && !dummyIsOptional &&
      evaluate::IsNullPointer(&actual)) {
    messages.Say(
        "Actual argument associated with %s may not be null pointer %s"_err_en_US,
        dummyName, actual.AsFortran());
  }

  // Warn about dubious actual argument association with a TARGET dummy
  // argument
  if (dummy.attrs.test(characteristics::DummyDataObject::Attr::Target) &&
      context.ShouldWarn(common::UsageWarning::NonTargetPassedToTarget)) {
    bool actualIsVariable{evaluate::IsVariable(actual)};
    bool actualIsTemp{
        !actualIsVariable || HasVectorSubscript(actual) || actualCoarrayRef};
    if (actualIsTemp) {
      messages.Say(common::UsageWarning::NonTargetPassedToTarget,
          "Any pointer associated with TARGET %s during this call will not be associated with the value of '%s' afterwards"_warn_en_US,
          dummyName, actual.AsFortran());
    } else {
      auto actualSymbolVector{GetSymbolVector(actual)};
      if (!evaluate::GetLastTarget(actualSymbolVector)) {
        messages.Say(common::UsageWarning::NonTargetPassedToTarget,
            "Any pointer associated with TARGET %s during this call must not be used afterwards, as '%s' is not a target"_warn_en_US,
            dummyName, actual.AsFortran());
      }
    }
  }

  // CUDA specific checks
  // TODO: These are disabled in OpenACC constructs, which may not be
  // correct when the target is not a GPU.
  if (!intrinsic &&
      !dummy.attrs.test(characteristics::DummyDataObject::Attr::Value) &&
      !FindOpenACCConstructContaining(scope)) {
    std::optional<common::CUDADataAttr> actualDataAttr, dummyDataAttr;
    if (const auto *actualObject{actualLastSymbol
                ? actualLastSymbol->detailsIf<ObjectEntityDetails>()
                : nullptr}) {
      actualDataAttr = actualObject->cudaDataAttr();
    }
    dummyDataAttr = dummy.cudaDataAttr;
    // Treat MANAGED like DEVICE for nonallocatable nonpointer arguments to
    // device subprograms
    if (procedure.cudaSubprogramAttrs.value_or(
            common::CUDASubprogramAttrs::Host) !=
            common::CUDASubprogramAttrs::Host &&
        !dummy.attrs.test(
            characteristics::DummyDataObject::Attr::Allocatable) &&
        !dummy.attrs.test(characteristics::DummyDataObject::Attr::Pointer)) {
      if (!dummyDataAttr || *dummyDataAttr == common::CUDADataAttr::Managed) {
        dummyDataAttr = common::CUDADataAttr::Device;
      }
      if ((!actualDataAttr && FindCUDADeviceContext(scope)) ||
          (actualDataAttr &&
              *actualDataAttr == common::CUDADataAttr::Managed)) {
        actualDataAttr = common::CUDADataAttr::Device;
      }
      // For device procedures, treat actual arguments with VALUE attribute as
      // device data
      if (!actualDataAttr && actualLastSymbol && IsValue(*actualLastSymbol) &&
          (*procedure.cudaSubprogramAttrs ==
              common::CUDASubprogramAttrs::Device)) {
        actualDataAttr = common::CUDADataAttr::Device;
      }
    }
    if (dummyDataAttr == common::CUDADataAttr::Device &&
        (dummyIsAssumedShape || dummyIsAssumedRank) &&
        !dummy.ignoreTKR.test(common::IgnoreTKR::Contiguous)) {
      if (auto contig{evaluate::IsContiguous(actual, foldingContext,
              /*namedConstantSectionsAreContiguous=*/true,
              /*firstDimensionStride1=*/true)}) {
        if (!*contig) {
          messages.Say(
              "actual argument associated with assumed shape/rank device %s is known to be discontiguous on its first dimension"_err_en_US,
              dummyName);
        }
      } else {
        messages.Say(
            "actual argument associated with assumed shape/rank device %s is not known to be contiguous on its first dimension"_warn_en_US,
            dummyName);
      }
    }
    bool isHostDeviceProc{procedure.cudaSubprogramAttrs &&
        *procedure.cudaSubprogramAttrs ==
            common::CUDASubprogramAttrs::HostDevice};
    if (!common::AreCompatibleCUDADataAttrs(dummyDataAttr, actualDataAttr,
            dummy.ignoreTKR, /*allowUnifiedMatchingRule=*/true,
            isHostDeviceProc, &context.languageFeatures())) {
      auto toStr{[](std::optional<common::CUDADataAttr> x) {
        return x ? "ATTRIBUTES("s +
                parser::ToUpperCaseLetters(common::EnumToString(*x)) + ")"s
                 : "no CUDA data attribute"s;
      }};
      messages.Say(
          "%s has %s but its associated actual argument has %s"_err_en_US,
          dummyName, toStr(dummyDataAttr), toStr(actualDataAttr));
    }
  }

  // Warning for breaking F'2023 change with character allocatables
  if (intrinsic && dummy.intent != common::Intent::In) {
    WarnOnDeferredLengthCharacterScalar(
        context, &actual, messages.at(), dummyName.c_str());
  }

  // %VAL() and %REF() checking for explicit interface
  if ((arg.isPercentRef() || arg.isPercentVal()) &&
      dummy.IsPassedByDescriptor(procedure.IsBindC())) {
    messages.Say(
        "%%VAL or %%REF are not allowed for %s that must be passed by means of a descriptor"_err_en_US,
        dummyName);
  }
  if (arg.isPercentVal() &&
      (!actualType.type().IsLengthlessIntrinsicType() ||
          actualType.Rank() != 0)) {
    messages.Say(
        "%VAL argument must be a scalar numeric or logical expression"_err_en_US);
  }
}

static void CheckProcedureArg(evaluate::ActualArgument &arg,
    const characteristics::Procedure &proc,
    const characteristics::DummyProcedure &dummy, const std::string &dummyName,
    SemanticsContext &context, bool ignoreImplicitVsExplicit) {
  evaluate::FoldingContext &foldingContext{context.foldingContext()};
  parser::ContextualMessages &messages{foldingContext.messages()};
  parser::CharBlock location{arg.sourceLocation().value_or(messages.at())};
  auto restorer{messages.SetLocation(location)};
  const characteristics::Procedure &interface { dummy.procedure.value() };
  if (const auto *expr{arg.UnwrapExpr()}) {
    bool dummyIsPointer{
        dummy.attrs.test(characteristics::DummyProcedure::Attr::Pointer)};
    const auto *argProcDesignator{
        std::get_if<evaluate::ProcedureDesignator>(&expr->u)};
    const auto *argProcSymbol{
        argProcDesignator ? argProcDesignator->GetSymbol() : nullptr};
    if (argProcSymbol) {
      if (const auto *subp{
              argProcSymbol->GetUltimate().detailsIf<SubprogramDetails>()}) {
        if (subp->stmtFunction()) {
          evaluate::SayWithDeclaration(messages, *argProcSymbol,
              "Statement function '%s' may not be passed as an actual argument"_err_en_US,
              argProcSymbol->name());
          return;
        }
      } else if (argProcSymbol->has<ProcBindingDetails>()) {
        if (!context.IsEnabled(common::LanguageFeature::BindingAsProcedure)) {
          evaluate::SayWithDeclaration(messages, *argProcSymbol,
              "Procedure binding '%s' passed as an actual argument"_err_en_US,
              argProcSymbol->name());
        } else if (context.ShouldWarn(
                       common::LanguageFeature::BindingAsProcedure)) {
          evaluate::SayWithDeclaration(messages, *argProcSymbol,
              common::LanguageFeature::BindingAsProcedure,
              "Procedure binding '%s' passed as an actual argument"_port_en_US,
              argProcSymbol->name());
        }
      }
    }
    if (auto argChars{characteristics::DummyArgument::FromActual(
            "actual argument", *expr, foldingContext,
            /*forImplicitInterface=*/true)}) {
      if (!argChars->IsTypelessIntrinsicDummy()) {
        if (auto *argProc{
                std::get_if<characteristics::DummyProcedure>(&argChars->u)}) {
          characteristics::Procedure &argInterface{argProc->procedure.value()};
          argInterface.attrs.reset(
              characteristics::Procedure::Attr::NullPointer);
          argInterface.attrs.reset(
              characteristics::Procedure::Attr::NullAllocatable);
          if (!argProcSymbol || argProcSymbol->attrs().test(Attr::INTRINSIC)) {
            // It's ok to pass ELEMENTAL unrestricted intrinsic functions.
            argInterface.attrs.reset(
                characteristics::Procedure::Attr::Elemental);
          } else if (argInterface.attrs.test(
                         characteristics::Procedure::Attr::Elemental)) {
            if (argProcSymbol) { // C1533
              evaluate::SayWithDeclaration(messages, *argProcSymbol,
                  "Non-intrinsic ELEMENTAL procedure '%s' may not be passed as an actual argument"_err_en_US,
                  argProcSymbol->name());
              return; // avoid piling on with checks below
            } else {
              argInterface.attrs.reset(
                  characteristics::Procedure::Attr::NullPointer);
              argInterface.attrs.reset(
                  characteristics::Procedure::Attr::NullAllocatable);
            }
          }
          if (interface.HasExplicitInterface()) {
            std::string whyNot;
            std::optional<std::string> warning;
            if (!interface.IsCompatibleWith(argInterface,
                    ignoreImplicitVsExplicit, &whyNot,
                    /*specificIntrinsic=*/nullptr, &warning)) {
              // 15.5.2.9(1): Explicit interfaces must match
              if (argInterface.HasExplicitInterface()) {
                messages.Say(
                    "Actual procedure argument has interface incompatible with %s: %s"_err_en_US,
                    dummyName, whyNot);
                return;
              } else if (proc.IsPure()) {
                messages.Say(
                    "Actual procedure argument for %s of a PURE procedure must have an explicit interface"_err_en_US,
                    dummyName);
              } else if (context.ShouldWarn(
                             common::UsageWarning::ImplicitInterfaceActual)) {
                messages.Say(common::UsageWarning::ImplicitInterfaceActual,
                    "Actual procedure argument has an implicit interface which is not known to be compatible with %s which has an explicit interface"_warn_en_US,
                    dummyName);
              }
            } else if (warning &&
                context.ShouldWarn(common::UsageWarning::ProcDummyArgShapes)) {
              messages.Say(common::UsageWarning::ProcDummyArgShapes,
                  "Actual procedure argument has possible interface incompatibility with %s: %s"_warn_en_US,
                  dummyName, std::move(*warning));
            }
          } else { // 15.5.2.9(2,3)
            if (interface.IsSubroutine() && argInterface.IsFunction()) {
              messages.Say(
                  "Actual argument associated with procedure %s is a function but must be a subroutine"_err_en_US,
                  dummyName);
            } else if (interface.IsFunction()) {
              if (argInterface.IsFunction()) {
                std::string whyNot;
                if (!interface.functionResult->IsCompatibleWith(
                        *argInterface.functionResult, &whyNot)) {
                  messages.Say(
                      "Actual argument function associated with procedure %s is not compatible: %s"_err_en_US,
                      dummyName, whyNot);
                }
              } else if (argInterface.IsSubroutine()) {
                messages.Say(
                    "Actual argument associated with procedure %s is a subroutine but must be a function"_err_en_US,
                    dummyName);
              }
            }
          }
        } else {
          messages.Say(
              "Actual argument associated with procedure %s is not a procedure"_err_en_US,
              dummyName);
        }
      } else if (IsNullPointer(expr)) {
        if (!dummyIsPointer &&
            !dummy.attrs.test(
                characteristics::DummyProcedure::Attr::Optional)) {
          messages.Say(
              "Actual argument associated with procedure %s is a null pointer"_err_en_US,
              dummyName);
        }
      } else {
        messages.Say(
            "Actual argument associated with procedure %s is typeless"_err_en_US,
            dummyName);
      }
    }
    if (dummyIsPointer) {
      if (dummy.intent == common::Intent::In) {
        // need not be definable, can be a target
      } else if (!IsProcedurePointer(*expr)) {
        messages.Say(
            "Actual argument associated with procedure pointer %s is not a procedure pointer"_err_en_US,
            dummyName);
      } else if (dummy.intent == common::Intent::Default) {
        // ok, needs to be definable only if defined at run time
      } else {
        DefinabilityFlags flags{DefinabilityFlag::PointerDefinition};
        if (dummy.intent != common::Intent::Out) {
          flags.set(DefinabilityFlag::DoNotNoteDefinition);
        }
        if (auto whyNot{WhyNotDefinable(
                location, context.FindScope(location), flags, *expr)}) {
          if (auto *msg{messages.Say(
                  "Actual argument associated with INTENT(%s) procedure pointer %s is not definable"_err_en_US,
                  dummy.intent == common::Intent::Out ? "OUT" : "IN OUT",
                  dummyName)}) {
            msg->Attach(
                std::move(whyNot->set_severity(parser::Severity::Because)));
          }
        }
      }
    }
  } else {
    messages.Say(
        "Assumed-type argument may not be forwarded as procedure %s"_err_en_US,
        dummyName);
  }
}

// Allow BOZ literal actual arguments when they can be converted to a known
// dummy argument type
static void ConvertBOZLiteralArg(
    evaluate::ActualArgument &arg, const evaluate::DynamicType &type) {
  if (auto *expr{arg.UnwrapExpr()}) {
    if (IsBOZLiteral(*expr)) {
      if (auto converted{evaluate::ConvertToType(type, SomeExpr{*expr})}) {
        arg = std::move(*converted);
      }
    }
  }
}

static void CheckExplicitInterfaceArg(evaluate::ActualArgument &arg,
    const characteristics::DummyArgument &dummy,
    const characteristics::Procedure &proc, SemanticsContext &context,
    const Scope *scope, const evaluate::SpecificIntrinsic *intrinsic,
    bool allowActualArgumentConversions, bool extentErrors,
    bool ignoreImplicitVsExplicit) {
  evaluate::FoldingContext &foldingContext{context.foldingContext()};
  auto &messages{foldingContext.messages()};
  std::string dummyName{"dummy argument"};
  if (!dummy.name.empty()) {
    dummyName += " '"s + parser::ToLowerCaseLetters(dummy.name) + "='";
  }
  auto restorer{
      messages.SetLocation(arg.sourceLocation().value_or(messages.at()))};
  auto CheckActualArgForLabel = [&](evaluate::ActualArgument &arg) {
    if (arg.isAlternateReturn()) {
      messages.Say(
          "Alternate return label '%d' cannot be associated with %s"_err_en_US,
          arg.GetLabel(), dummyName);
      return false;
    } else {
      return true;
    }
  };
  common::visit(
      common::visitors{
          [&](const characteristics::DummyDataObject &object) {
            if (CheckActualArgForLabel(arg)) {
              ConvertBOZLiteralArg(arg, object.type.type());
              if (auto *expr{arg.UnwrapExpr()}) {
                if (auto type{characteristics::TypeAndShape::Characterize(
                        *expr, foldingContext)}) {
                  arg.set_dummyIntent(object.intent);
                  bool isElemental{
                      object.type.Rank() == 0 && proc.IsElemental()};
                  CheckExplicitDataArg(object, dummyName, *expr, *type,
                      isElemental, context, foldingContext, scope, intrinsic,
                      allowActualArgumentConversions, extentErrors, proc, arg);
                } else if (object.type.type().IsTypelessIntrinsicArgument() &&
                    IsBOZLiteral(*expr)) {
                  // ok
                } else if (object.type.type().IsTypelessIntrinsicArgument() &&
                    evaluate::IsNullObjectPointer(expr)) {
                  // ok, ASSOCIATED(NULL(without MOLD=))
                } else if (object.type.attrs().test(characteristics::
                                   TypeAndShape::Attr::AssumedRank) &&
                    evaluate::IsNullObjectPointer(expr) &&
                    (object.attrs.test(
                         characteristics::DummyDataObject::Attr::Allocatable) ||
                        object.attrs.test(
                            characteristics::DummyDataObject::Attr::Pointer) ||
                        !object.attrs.test(characteristics::DummyDataObject::
                                Attr::Optional))) {
                  messages.Say(
                      "NULL() without MOLD= must not be associated with an assumed-rank dummy argument that is ALLOCATABLE, POINTER, or non-OPTIONAL"_err_en_US);
                } else if ((object.attrs.test(characteristics::DummyDataObject::
                                    Attr::Pointer) ||
                               object.attrs.test(characteristics::
                                       DummyDataObject::Attr::Optional)) &&
                    evaluate::IsNullObjectPointer(expr)) {
                  // FOO(NULL(without MOLD=))
                  if (object.type.type().IsAssumedLengthCharacter()) {
                    messages.Say(
                        "Actual argument associated with %s is a NULL() pointer without a MOLD= to provide a character length"_err_en_US,
                        dummyName);
                  } else if (const DerivedTypeSpec *
                      derived{GetDerivedTypeSpec(object.type.type())}) {
                    for (const auto &[pName, pValue] : derived->parameters()) {
                      if (pValue.isAssumed()) {
                        messages.Say(
                            "Actual argument associated with %s is a NULL() pointer without a MOLD= to provide a value for the assumed type parameter '%s'"_err_en_US,
                            dummyName, pName.ToString());
                        break;
                      }
                    }
                  }
                } else if (object.attrs.test(characteristics::DummyDataObject::
                                   Attr::Allocatable) &&
                    (evaluate::IsNullAllocatable(expr) ||
                        evaluate::IsBareNullPointer(expr))) {
                  if (object.intent == common::Intent::Out ||
                      object.intent == common::Intent::InOut) {
                    messages.Say(
                        "NULL() actual argument '%s' may not be associated with allocatable dummy argument %s that is INTENT(OUT) or INTENT(IN OUT)"_err_en_US,
                        expr->AsFortran(), dummyName);
                  } else if (object.intent == common::Intent::Default &&
                      context.ShouldWarn(common::UsageWarning::
                              NullActualForDefaultIntentAllocatable)) {
                    messages.Say(common::UsageWarning::
                                     NullActualForDefaultIntentAllocatable,
                        "NULL() actual argument '%s' should not be associated with allocatable dummy argument %s without INTENT(IN)"_warn_en_US,
                        expr->AsFortran(), dummyName);
                  } else if (context.ShouldWarn(common::LanguageFeature::
                                     NullActualForAllocatable)) {
                    messages.Say(
                        common::LanguageFeature::NullActualForAllocatable,
                        "Allocatable %s is associated with %s"_port_en_US,
                        dummyName, expr->AsFortran());
                  }
                } else {
                  messages.Say(
                      "Actual argument '%s' associated with %s is not a variable or typed expression"_err_en_US,
                      expr->AsFortran(), dummyName);
                }
              } else {
                const Symbol &assumed{DEREF(arg.GetAssumedTypeDummy())};
                if (!object.type.type().IsAssumedType()) {
                  messages.Say(
                      "Assumed-type '%s' may be associated only with an assumed-type %s"_err_en_US,
                      assumed.name(), dummyName);
                } else if (object.type.attrs().test(characteristics::
                                   TypeAndShape::Attr::AssumedRank) &&
                    !IsAssumedShape(assumed) && !IsAssumedRank(assumed)) {
                  messages.Say( // C711
                      "Assumed-type '%s' must be either assumed shape or assumed rank to be associated with assumed rank %s"_err_en_US,
                      assumed.name(), dummyName);
                }
              }
            }
          },
          [&](const characteristics::DummyProcedure &dummy) {
            if (CheckActualArgForLabel(arg)) {
              CheckProcedureArg(arg, proc, dummy, dummyName, context,
                  ignoreImplicitVsExplicit);
            }
          },
          [&](const characteristics::AlternateReturn &) {
            // All semantic checking is done elsewhere
          },
      },
      dummy.u);
}

static void RearrangeArguments(const characteristics::Procedure &proc,
    evaluate::ActualArguments &actuals, parser::ContextualMessages &messages) {
  CHECK(proc.HasExplicitInterface());
  if (actuals.size() < proc.dummyArguments.size()) {
    actuals.resize(proc.dummyArguments.size());
  } else if (actuals.size() > proc.dummyArguments.size()) {
    messages.Say(
        "Too many actual arguments (%zd) passed to procedure that expects only %zd"_err_en_US,
        actuals.size(), proc.dummyArguments.size());
  }
  std::map<std::string, evaluate::ActualArgument> kwArgs;
  bool anyKeyword{false};
  int which{1};
  for (auto &x : actuals) {
    if (!x) {
    } else if (x->keyword()) {
      auto emplaced{
          kwArgs.try_emplace(x->keyword()->ToString(), std::move(*x))};
      if (!emplaced.second) {
        messages.Say(*x->keyword(),
            "Argument keyword '%s=' appears on more than one effective argument in this procedure reference"_err_en_US,
            *x->keyword());
      }
      x.reset();
      anyKeyword = true;
    } else if (anyKeyword) {
      messages.Say(x ? x->sourceLocation() : std::nullopt,
          "Actual argument #%d without a keyword may not follow any actual argument with a keyword"_err_en_US,
          which);
    }
    ++which;
  }
  if (!kwArgs.empty()) {
    int index{0};
    for (const auto &dummy : proc.dummyArguments) {
      if (!dummy.name.empty()) {
        auto iter{kwArgs.find(dummy.name)};
        if (iter != kwArgs.end()) {
          evaluate::ActualArgument &x{iter->second};
          if (actuals[index]) {
            messages.Say(*x.keyword(),
                "Keyword argument '%s=' has already been specified positionally (#%d) in this procedure reference"_err_en_US,
                *x.keyword(), index + 1);
          } else {
            actuals[index] = std::move(x);
          }
          kwArgs.erase(iter);
        }
      }
      ++index;
    }
    for (auto &bad : kwArgs) {
      evaluate::ActualArgument &x{bad.second};
      messages.Say(*x.keyword(),
          "Argument keyword '%s=' is not recognized for this procedure reference"_err_en_US,
          *x.keyword());
    }
  }
}

// 15.8.1(3) -- In a reference to an elemental procedure, if any argument is an
// array, each actual argument that corresponds to an INTENT(OUT) or
// INTENT(INOUT) dummy argument shall be an array. The actual argument to an
// ELEMENTAL procedure must conform.
static bool CheckElementalConformance(parser::ContextualMessages &messages,
    const characteristics::Procedure &proc, evaluate::ActualArguments &actuals,
    evaluate::FoldingContext &context) {
  std::optional<evaluate::Shape> shape;
  std::string shapeName;
  int index{0};
  bool hasArrayArg{false};
  for (const auto &arg : actuals) {
    if (arg && !arg->isAlternateReturn() && arg->Rank() > 0) {
      hasArrayArg = true;
      break;
    }
  }
  for (const auto &arg : actuals) {
    const auto &dummy{proc.dummyArguments.at(index++)};
    if (arg) {
      if (const auto *expr{arg->UnwrapExpr()}) {
        if (const auto *wholeSymbol{evaluate::UnwrapWholeSymbolDataRef(arg)}) {
          wholeSymbol = &ResolveAssociations(*wholeSymbol);
          if (IsAssumedSizeArray(*wholeSymbol)) {
            evaluate::SayWithDeclaration(messages, *wholeSymbol,
                "Whole assumed-size array '%s' may not be used as an argument to an elemental procedure"_err_en_US,
                wholeSymbol->name());
          }
        }
        if (auto argShape{evaluate::GetShape(context, *expr)}) {
          if (GetRank(*argShape) > 0) {
            std::string argName{"actual argument ("s + expr->AsFortran() +
                ") corresponding to dummy argument #" + std::to_string(index) +
                " ('" + dummy.name + "')"};
            if (shape) {
              auto tristate{evaluate::CheckConformance(messages, *shape,
                  *argShape, evaluate::CheckConformanceFlags::None,
                  shapeName.c_str(), argName.c_str())};
              if (tristate && !*tristate) {
                return false;
              }
            } else {
              shape = std::move(argShape);
              shapeName = argName;
            }
          } else if ((dummy.GetIntent() == common::Intent::Out ||
                         dummy.GetIntent() == common::Intent::InOut) &&
              hasArrayArg) {
            messages.Say(
                "In an elemental procedure reference with at least one array argument, actual argument %s that corresponds to an INTENT(OUT) or INTENT(INOUT) dummy argument must be an array"_err_en_US,
                expr->AsFortran());
          }
        }
      }
    }
  }
  return true;
}

// ASSOCIATED (16.9.16)
static void CheckAssociated(evaluate::ActualArguments &arguments,
    SemanticsContext &semanticsContext, const Scope *scope) {
  evaluate::FoldingContext &foldingContext{semanticsContext.foldingContext()};
  parser::ContextualMessages &messages{foldingContext.messages()};
  bool ok{true};
  if (arguments.size() < 2) {
    return;
  }
  if (const auto &pointerArg{arguments[0]}) {
    if (const auto *pointerExpr{pointerArg->UnwrapExpr()}) {
      if (!IsPointer(*pointerExpr)) {
        messages.Say(pointerArg->sourceLocation(),
            "POINTER= argument of ASSOCIATED() must be a pointer"_err_en_US);
        return;
      }
      if (const auto &targetArg{arguments[1]}) {
        // The standard requires that the TARGET= argument, when present,
        // be type compatible with the POINTER= for a data pointer.  In
        // the case of procedure pointers, the standard requires that it
        // be a valid RHS for a pointer assignment that has the POINTER=
        // argument as its LHS.  Some popular compilers misinterpret this
        // requirement more strongly than necessary, and actually validate
        // the POINTER= argument as if it were serving as the LHS of a pointer
        // assignment.  This, perhaps unintentionally, excludes function
        // results, including NULL(), from being used there, as well as
        // INTENT(IN) dummy pointers.  Detect these conditions and emit
        // portability warnings.
        if (semanticsContext.ShouldWarn(common::UsageWarning::Portability)) {
          if (!evaluate::ExtractDataRef(*pointerExpr) &&
              !evaluate::IsProcedurePointer(*pointerExpr)) {
            messages.Say(common::UsageWarning::Portability,
                pointerArg->sourceLocation(),
                "POINTER= argument of ASSOCIATED() is required by some other compilers to be a pointer"_port_en_US);
          } else if (scope && !evaluate::UnwrapProcedureRef(*pointerExpr)) {
            if (auto whyNot{WhyNotDefinable(
                    pointerArg->sourceLocation().value_or(messages.at()),
                    *scope,
                    DefinabilityFlags{DefinabilityFlag::PointerDefinition,
                        DefinabilityFlag::DoNotNoteDefinition},
                    *pointerExpr)}) {
              if (whyNot->IsFatal()) {
                if (auto *msg{messages.Say(common::UsageWarning::Portability,
                        pointerArg->sourceLocation(),
                        "POINTER= argument of ASSOCIATED() is required by some other compilers to be a valid left-hand side of a pointer assignment statement"_port_en_US)}) {
                  msg->Attach(std::move(
                      whyNot->set_severity(parser::Severity::Because)));
                }
              } else {
                messages.Say(std::move(*whyNot));
              }
            }
          }
        }
        if (const auto *targetExpr{targetArg->UnwrapExpr()}) {
          if (IsProcedurePointer(*pointerExpr) &&
              !IsBareNullPointer(pointerExpr)) { // POINTER= is a procedure
            if (auto pointerProc{characteristics::Procedure::Characterize(
                    *pointerExpr, foldingContext)}) {
              if (IsBareNullPointer(targetExpr)) {
              } else if (IsProcedurePointerTarget(*targetExpr)) {
                if (auto targetProc{characteristics::Procedure::Characterize(
                        *targetExpr, foldingContext)}) {
                  bool isCall{!!UnwrapProcedureRef(*targetExpr)};
                  std::string whyNot;
                  std::optional<std::string> warning;
                  const auto *targetProcDesignator{
                      evaluate::UnwrapExpr<evaluate::ProcedureDesignator>(
                          *targetExpr)};
                  const evaluate::SpecificIntrinsic *specificIntrinsic{
                      targetProcDesignator
                          ? targetProcDesignator->GetSpecificIntrinsic()
                          : nullptr};
                  std::optional<parser::MessageFixedText> msg{
                      CheckProcCompatibility(isCall, pointerProc, &*targetProc,
                          specificIntrinsic, whyNot, warning,
                          /*ignoreImplicitVsExplicit=*/false)};
                  std::optional<common::UsageWarning> whichWarning;
                  if (!msg && warning &&
                      semanticsContext.ShouldWarn(
                          common::UsageWarning::ProcDummyArgShapes)) {
                    whichWarning = common::UsageWarning::ProcDummyArgShapes;
                    msg =
                        "Procedures '%s' and '%s' may not be completely compatible: %s"_warn_en_US;
                    whyNot = std::move(*warning);
                  } else if (msg && !msg->IsFatal() &&
                      semanticsContext.ShouldWarn(
                          common::UsageWarning::ProcPointerCompatibility)) {
                    whichWarning =
                        common::UsageWarning::ProcPointerCompatibility;
                  }
                  if (msg && (msg->IsFatal() || whichWarning)) {
                    if (auto *said{messages.Say(std::move(*msg),
                            "pointer '" + pointerExpr->AsFortran() + "'",
                            targetExpr->AsFortran(), whyNot)};
                        said && whichWarning) {
                      said->set_usageWarning(*whichWarning);
                    }
                  }
                }
              } else if (!IsNullProcedurePointer(targetExpr)) {
                messages.Say(
                    "POINTER= argument '%s' is a procedure pointer but the TARGET= argument '%s' is not a procedure or procedure pointer"_err_en_US,
                    pointerExpr->AsFortran(), targetExpr->AsFortran());
              }
            }
          } else if (IsVariable(*targetExpr) || IsNullPointer(targetExpr)) {
            // Object pointer and target
            if (ExtractDataRef(*targetExpr)) {
              if (SymbolVector symbols{GetSymbolVector(*targetExpr)};
                  !evaluate::GetLastTarget(symbols)) {
                parser::Message *msg{messages.Say(targetArg->sourceLocation(),
                    "TARGET= argument '%s' must have either the POINTER or the TARGET attribute"_err_en_US,
                    targetExpr->AsFortran())};
                for (SymbolRef ref : symbols) {
                  msg = evaluate::AttachDeclaration(msg, *ref);
                }
              } else if (HasVectorSubscript(*targetExpr) ||
                  ExtractCoarrayRef(*targetExpr)) {
                messages.Say(targetArg->sourceLocation(),
                    "TARGET= argument '%s' may not have a vector subscript or coindexing"_err_en_US,
                    targetExpr->AsFortran());
              }
            }
            if (const auto pointerType{pointerArg->GetType()}) {
              if (const auto targetType{targetArg->GetType()}) {
                ok = pointerType->IsTkCompatibleWith(*targetType) ||
                    targetType->IsTkCompatibleWith(*pointerType);
              }
            }
          } else {
            messages.Say(
                "POINTER= argument '%s' is an object pointer but the TARGET= argument '%s' is not a variable"_err_en_US,
                pointerExpr->AsFortran(), targetExpr->AsFortran());
          }
          if (!IsAssumedRank(*pointerExpr)) {
            if (IsAssumedRank(*targetExpr)) {
              messages.Say(
                  "TARGET= argument '%s' may not be assumed-rank when POINTER= argument is not"_err_en_US,
                  pointerExpr->AsFortran());
            } else if (pointerExpr->Rank() != targetExpr->Rank()) {
              messages.Say(
                  "POINTER= argument and TARGET= argument have incompatible ranks %d and %d"_err_en_US,
                  pointerExpr->Rank(), targetExpr->Rank());
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
    messages.Say(
        "Arguments of ASSOCIATED() must be a pointer and an optional valid target"_err_en_US);
  }
}

// CO_REDUCE (F'2023 16.9.49)
static void CheckCoReduce(
    evaluate::ActualArguments &arguments, evaluate::FoldingContext &context) {
  parser::ContextualMessages &messages{context.messages()};
  evaluate::CheckForCoindexedObject(
      context.messages(), arguments[0], "co_reduce", "a");
  evaluate::CheckForCoindexedObject(
      context.messages(), arguments[2], "co_reduce", "stat");
  evaluate::CheckForCoindexedObject(
      context.messages(), arguments[3], "co_reduce", "errmsg");

  std::optional<evaluate::DynamicType> aType;
  if (const auto &a{arguments[0]}) {
    aType = a->GetType();
  }
  std::optional<characteristics::Procedure> procChars;
  if (const auto &operation{arguments[1]}) {
    if (const auto *expr{operation->UnwrapExpr()}) {
      if (const auto *designator{
              std::get_if<evaluate::ProcedureDesignator>(&expr->u)}) {
        procChars = characteristics::Procedure::Characterize(
            *designator, context, /*emitError=*/true);
      } else if (const auto *ref{
                     std::get_if<evaluate::ProcedureRef>(&expr->u)}) {
        procChars = characteristics::Procedure::Characterize(*ref, context);
      }
    }
  }

  static constexpr characteristics::DummyDataObject::Attrs notAllowedArgAttrs{
      characteristics::DummyDataObject::Attr::Optional,
      characteristics::DummyDataObject::Attr::Allocatable,
      characteristics::DummyDataObject::Attr::Pointer,
  };
  static constexpr characteristics::FunctionResult::Attrs
      notAllowedFuncResAttrs{
          characteristics::FunctionResult::Attr::Allocatable,
          characteristics::FunctionResult::Attr::Pointer,
      };
  const characteristics::TypeAndShape *result{
      procChars && procChars->functionResult
          ? procChars->functionResult->GetTypeAndShape()
          : nullptr};
  if (!procChars || !procChars->IsPure() ||
      procChars->dummyArguments.size() != 2 || !procChars->functionResult) {
    messages.Say(
        "OPERATION= argument of CO_REDUCE() must be a pure function of two data arguments"_err_en_US);
  } else if (procChars->attrs.test(characteristics::Procedure::Attr::BindC)) {
    messages.Say(
        "A BIND(C) OPERATION= argument of CO_REDUCE() is not supported"_err_en_US);
  } else if (!result || result->Rank() != 0) {
    messages.Say(
        "OPERATION= argument of CO_REDUCE() must be a scalar function"_err_en_US);
  } else if (result->type().IsPolymorphic() ||
      (aType && !aType->IsTkLenCompatibleWith(result->type()))) {
    messages.Say(
        "OPERATION= argument of CO_REDUCE() must have the same type as A="_err_en_US);
  } else if (((procChars->functionResult->attrs & notAllowedFuncResAttrs) !=
                 characteristics::FunctionResult::Attrs{}) ||
      procChars->functionResult->GetTypeAndShape()->type().IsPolymorphic()) {
    messages.Say(
        "Result of OPERATION= procedure of CO_REDUCE() must be scalar and neither allocatable, pointer, nor polymorphic"_err_en_US);
  } else {
    const characteristics::DummyDataObject *data[2]{};
    for (int j{0}; j < 2; ++j) {
      const auto &dummy{procChars->dummyArguments.at(j)};
      data[j] = std::get_if<characteristics::DummyDataObject>(&dummy.u);
    }
    if (!data[0] || !data[1]) {
      messages.Say(
          "OPERATION= argument of CO_REDUCE() may not have dummy procedure arguments"_err_en_US);
    } else {
      for (int j{0}; j < 2; ++j) {
        if (((data[j]->attrs & notAllowedArgAttrs) !=
                characteristics::DummyDataObject::Attrs{}) ||
            data[j]->type.Rank() != 0 || data[j]->type.type().IsPolymorphic() ||
            (aType && !data[j]->type.type().IsTkCompatibleWith(*aType))) {
          messages.Say(
              "Arguments of OPERATION= procedure of CO_REDUCE() must be both scalar of the same type as A=, and neither allocatable, pointer, polymorphic, nor optional"_err_en_US);
          break;
        }
      }
      static constexpr characteristics::DummyDataObject::Attrs attrs{
          characteristics::DummyDataObject::Attr::Asynchronous,
          characteristics::DummyDataObject::Attr::Target,
          characteristics::DummyDataObject::Attr::Value,
      };
      if ((data[0]->attrs & attrs) != (data[1]->attrs & attrs)) {
        messages.Say(
            "If either argument of the OPERATION= procedure of CO_REDUCE() has the ASYNCHRONOUS, TARGET, or VALUE attribute, both must have that attribute"_err_en_US);
      }
    }
  }
}

// EVENT_QUERY (F'2023 16.9.82)
static void CheckEvent_Query(evaluate::ActualArguments &arguments,
    evaluate::FoldingContext &foldingContext) {
  if (arguments.size() > 0 && arguments[0] &&
      ExtractCoarrayRef(*arguments[0]).has_value()) {
    foldingContext.messages().Say(arguments[0]->sourceLocation(),
        "EVENT= argument to EVENT_QUERY must not be coindexed"_err_en_US);
  }
  if (arguments.size() > 1 && arguments[1]) {
    if (auto dyType{arguments[1]->GetType()}) {
      int defaultInt{
          foldingContext.defaults().GetDefaultKind(TypeCategory::Integer)};
      if (dyType->category() == TypeCategory::Integer &&
          dyType->kind() < defaultInt) {
        foldingContext.messages().Say(arguments[1]->sourceLocation(),
            "COUNT= argument to EVENT_QUERY must be an integer with kind >= %d"_err_en_US,
            defaultInt);
      }
    }
  }
  if (arguments.size() > 2 && arguments[2]) {
    if (auto dyType{arguments[2]->GetType()}) {
      if (dyType->category() == TypeCategory::Integer && dyType->kind() < 2) {
        foldingContext.messages().Say(arguments[2]->sourceLocation(),
            "STAT= argument to EVENT_QUERY must be an integer with kind >= 2 when present"_err_en_US);
      }
    }
  }
}

// IMAGE_INDEX (F'2023 16.9.107)
static void CheckImage_Index(evaluate::ActualArguments &arguments,
    parser::ContextualMessages &messages) {
  if (arguments[1] && arguments[0]) {
    if (const auto subArrShape{
            evaluate::GetShape(arguments[1]->UnwrapExpr())}) {
      if (const auto *coarrayArgSymbol{UnwrapWholeSymbolOrComponentDataRef(
              arguments[0]->UnwrapExpr())}) {
        auto coarrayArgCorank{coarrayArgSymbol->Corank()};
        if (auto subArrSize{evaluate::ToInt64(*subArrShape->front())}) {
          if (subArrSize != coarrayArgCorank) {
            messages.Say(arguments[1]->sourceLocation(),
                "The size of 'SUB=' (%jd) for intrinsic 'image_index' must be equal to the corank of 'COARRAY=' (%d)"_err_en_US,
                static_cast<std::int64_t>(*subArrSize), coarrayArgCorank);
          }
        }
      }
    }
  }
}

// Ensure that any optional argument that might be absent at run time
// does not require data conversion.
static void CheckMaxMin(const characteristics::Procedure &proc,
    evaluate::ActualArguments &arguments,
    parser::ContextualMessages &messages) {
  if (proc.functionResult) {
    if (const auto *typeAndShape{proc.functionResult->GetTypeAndShape()}) {
      for (std::size_t j{2}; j < arguments.size(); ++j) {
        if (arguments[j]) {
          if (const auto *expr{arguments[j]->UnwrapExpr()};
              expr && evaluate::MayBePassedAsAbsentOptional(*expr)) {
            if (auto thisType{expr->GetType()}) {
              if (thisType->category() == TypeCategory::Character &&
                  typeAndShape->type().category() == TypeCategory::Character &&
                  thisType->kind() == typeAndShape->type().kind()) {
                // don't care about lengths
              } else if (*thisType != typeAndShape->type()) {
                messages.Say(arguments[j]->sourceLocation(),
                    "An actual argument to MAX/MIN requiring data conversion may not be OPTIONAL, POINTER, or ALLOCATABLE"_err_en_US);
              }
            }
          }
        }
      }
    }
  }
}

static void CheckFree(evaluate::ActualArguments &arguments,
    parser::ContextualMessages &messages) {
  if (arguments.size() != 1) {
    messages.Say("FREE expects a single argument"_err_en_US);
  }
  auto arg = arguments[0];
  if (const Symbol * symbol{evaluate::UnwrapWholeSymbolDataRef(arg)};
      !symbol || !symbol->test(Symbol::Flag::CrayPointer)) {
    messages.Say("FREE should only be used with Cray pointers"_warn_en_US);
  }
}

// MOVE_ALLOC (F'2023 16.9.147)
static void CheckMove_Alloc(evaluate::ActualArguments &arguments,
    parser::ContextualMessages &messages) {
  if (arguments.size() >= 1) {
    evaluate::CheckForCoindexedObject(
        messages, arguments[0], "move_alloc", "from");
  }
  if (arguments.size() >= 2) {
    evaluate::CheckForCoindexedObject(
        messages, arguments[1], "move_alloc", "to");
    int fromCR{GetCorank(arguments[0])};
    int toCR{GetCorank(arguments[1])};
    if (fromCR != toCR) {
      messages.Say(*arguments[0]->sourceLocation(),
          "FROM= argument to MOVE_ALLOC has corank %d, but TO= argument has corank %d"_err_en_US,
          fromCR, toCR);
    }
  }
  if (arguments.size() >= 3) {
    evaluate::CheckForCoindexedObject(
        messages, arguments[2], "move_alloc", "stat");
  }
  if (arguments.size() >= 4) {
    evaluate::CheckForCoindexedObject(
        messages, arguments[3], "move_alloc", "errmsg");
  }
  if (arguments.size() >= 2 && arguments[0] && arguments[1]) {
    for (int j{0}; j < 2; ++j) {
      if (const Symbol *
              whole{UnwrapWholeSymbolOrComponentDataRef(arguments[j])};
          !whole || !IsAllocatable(whole->GetUltimate())) {
        messages.Say(*arguments[j]->sourceLocation(),
            "Argument #%d to MOVE_ALLOC must be allocatable"_err_en_US, j + 1);
      }
    }
    auto type0{arguments[0]->GetType()};
    auto type1{arguments[1]->GetType()};
    if (type0 && type1 && type0->IsPolymorphic() && !type1->IsPolymorphic()) {
      messages.Say(arguments[1]->sourceLocation(),
          "When MOVE_ALLOC(FROM=) is polymorphic, TO= must also be polymorphic"_err_en_US);
    }
  }
}

// PRESENT (F'2023 16.9.163)
static void CheckPresent(evaluate::ActualArguments &arguments,
    parser::ContextualMessages &messages) {
  if (arguments.size() == 1) {
    if (const auto &arg{arguments[0]}; arg) {
      const Symbol *symbol{nullptr};
      if (const auto *expr{arg->UnwrapExpr()}) {
        if (const auto *proc{
                std::get_if<evaluate::ProcedureDesignator>(&expr->u)}) {
          symbol = proc->GetSymbol();
        } else {
          symbol = evaluate::UnwrapWholeSymbolDataRef(*expr);
        }
      } else {
        symbol = arg->GetAssumedTypeDummy();
      }
      if (!symbol ||
          !symbol->GetUltimate().attrs().test(semantics::Attr::OPTIONAL)) {
        messages.Say(arg ? arg->sourceLocation() : messages.at(),
            "Argument of PRESENT() must be the name of a whole OPTIONAL dummy argument"_err_en_US);
      }
    }
  }
}

// REDUCE (F'2023 16.9.173)
static void CheckReduce(
    evaluate::ActualArguments &arguments, evaluate::FoldingContext &context) {
  std::optional<evaluate::DynamicType> arrayType;
  parser::ContextualMessages &messages{context.messages()};
  if (const auto &array{arguments[0]}) {
    arrayType = array->GetType();
    if (!arguments[/*identity=*/4]) {
      if (const auto *expr{array->UnwrapExpr()}) {
        if (auto shape{
                evaluate::GetShape(context, *expr, /*invariantOnly=*/false)}) {
          if (const auto &dim{arguments[2]}; dim && array->Rank() > 1) {
            // Partial reduction
            auto dimVal{evaluate::ToInt64(dim->UnwrapExpr())};
            std::int64_t j{0};
            int zeroDims{0};
            bool isSelectedDimEmpty{false};
            for (const auto &extent : *shape) {
              ++j;
              if (evaluate::ToInt64(extent) == 0) {
                ++zeroDims;
                isSelectedDimEmpty |= dimVal && j == *dimVal;
              }
            }
            if (isSelectedDimEmpty && zeroDims == 1) {
              messages.Say(
                  "IDENTITY= must be present when DIM=%d and the array has zero extent on that dimension"_err_en_US,
                  static_cast<int>(dimVal.value()));
            }
          } else { // no DIM= or DIM=1 on a vector: total reduction
            for (const auto &extent : *shape) {
              if (evaluate::ToInt64(extent) == 0) {
                messages.Say(
                    "IDENTITY= must be present when the array is empty and the result is scalar"_err_en_US);
                break;
              }
            }
          }
        }
      }
    }
  }
  std::optional<characteristics::Procedure> procChars;
  if (const auto &operation{arguments[1]}) {
    if (const auto *expr{operation->UnwrapExpr()}) {
      if (const auto *designator{
              std::get_if<evaluate::ProcedureDesignator>(&expr->u)}) {
        procChars = characteristics::Procedure::Characterize(
            *designator, context, /*emitError=*/true);
      } else if (const auto *ref{
                     std::get_if<evaluate::ProcedureRef>(&expr->u)}) {
        procChars = characteristics::Procedure::Characterize(*ref, context);
      }
    }
  }
  const auto *result{
      procChars ? procChars->functionResult->GetTypeAndShape() : nullptr};
  if (!procChars || !procChars->IsPure() ||
      procChars->dummyArguments.size() != 2 || !procChars->functionResult) {
    messages.Say(
        "OPERATION= argument of REDUCE() must be a pure function of two data arguments"_err_en_US);
  } else if (procChars->attrs.test(characteristics::Procedure::Attr::BindC)) {
    messages.Say(
        "A BIND(C) OPERATION= argument of REDUCE() is not supported"_err_en_US);
  } else if (!result || result->Rank() != 0) {
    messages.Say(
        "OPERATION= argument of REDUCE() must be a scalar function"_err_en_US);
  } else if (result->type().IsPolymorphic() ||
      (arrayType && !arrayType->IsTkLenCompatibleWith(result->type()))) {
    messages.Say(
        "OPERATION= argument of REDUCE() must have the same type as ARRAY="_err_en_US);
  } else {
    const characteristics::DummyDataObject *data[2]{};
    for (int j{0}; j < 2; ++j) {
      const auto &dummy{procChars->dummyArguments.at(j)};
      data[j] = std::get_if<characteristics::DummyDataObject>(&dummy.u);
    }
    if (!data[0] || !data[1]) {
      messages.Say(
          "OPERATION= argument of REDUCE() may not have dummy procedure arguments"_err_en_US);
    } else {
      for (int j{0}; j < 2; ++j) {
        if (data[j]->attrs.test(
                characteristics::DummyDataObject::Attr::Optional) ||
            data[j]->attrs.test(
                characteristics::DummyDataObject::Attr::Allocatable) ||
            data[j]->attrs.test(
                characteristics::DummyDataObject::Attr::Pointer) ||
            data[j]->type.Rank() != 0 || data[j]->type.type().IsPolymorphic() ||
            (arrayType &&
                !data[j]->type.type().IsTkCompatibleWith(*arrayType))) {
          messages.Say(
              "Arguments of OPERATION= procedure of REDUCE() must be both scalar of the same type as ARRAY=, and neither allocatable, pointer, polymorphic, nor optional"_err_en_US);
        }
      }
      static constexpr characteristics::DummyDataObject::Attr attrs[]{
          characteristics::DummyDataObject::Attr::Asynchronous,
          characteristics::DummyDataObject::Attr::Target,
          characteristics::DummyDataObject::Attr::Value,
      };
      for (std::size_t j{0}; j < sizeof attrs / sizeof *attrs; ++j) {
        if (data[0]->attrs.test(attrs[j]) != data[1]->attrs.test(attrs[j])) {
          messages.Say(
              "If either argument of the OPERATION= procedure of REDUCE() has the ASYNCHRONOUS, TARGET, or VALUE attribute, both must have that attribute"_err_en_US);
          break;
        }
      }
    }
  }
  // When the MASK= is present and has no .TRUE. element, and there is
  // no IDENTITY=, it's an error.
  if (const auto &mask{arguments[3]}; mask && !arguments[/*identity*/ 4]) {
    if (const auto *expr{mask->UnwrapExpr()}) {
      if (const auto *logical{
              std::get_if<evaluate::Expr<evaluate::SomeLogical>>(&expr->u)}) {
        if (common::visit(
                [](const auto &kindExpr) {
                  using KindExprType = std::decay_t<decltype(kindExpr)>;
                  using KindLogical = typename KindExprType::Result;
                  if (const auto *c{evaluate::UnwrapConstantValue<KindLogical>(
                          kindExpr)}) {
                    for (const auto &element : c->values()) {
                      if (element.IsTrue()) {
                        return false;
                      }
                    }
                    return true;
                  }
                  return false;
                },
                logical->u)) {
          messages.Say(
              "MASK= has no .TRUE. element, so IDENTITY= must be present"_err_en_US);
        }
      }
    }
  }
}

// TRANSFER (16.9.193)
static void CheckTransferOperandType(SemanticsContext &context,
    const evaluate::DynamicType &type, const char *which) {
  if (type.IsPolymorphic() &&
      context.ShouldWarn(common::UsageWarning::PolymorphicTransferArg)) {
    context.foldingContext().messages().Say(
        common::UsageWarning::PolymorphicTransferArg,
        "%s of TRANSFER is polymorphic"_warn_en_US, which);
  } else if (!type.IsUnlimitedPolymorphic() &&
      type.category() == TypeCategory::Derived &&
      context.ShouldWarn(common::UsageWarning::PointerComponentTransferArg)) {
    DirectComponentIterator directs{type.GetDerivedTypeSpec()};
    if (auto bad{std::find_if(directs.begin(), directs.end(), IsDescriptor)};
        bad != directs.end()) {
      evaluate::SayWithDeclaration(context.foldingContext().messages(), *bad,
          common::UsageWarning::PointerComponentTransferArg,
          "%s of TRANSFER contains allocatable or pointer component %s"_warn_en_US,
          which, bad.BuildResultDesignatorName());
    }
  }
}

static void CheckTransfer(evaluate::ActualArguments &arguments,
    SemanticsContext &context, const Scope *scope) {
  evaluate::FoldingContext &foldingContext{context.foldingContext()};
  parser::ContextualMessages &messages{foldingContext.messages()};
  if (arguments.size() >= 2) {
    if (auto source{characteristics::TypeAndShape::Characterize(
            arguments[0], foldingContext)}) {
      CheckTransferOperandType(context, source->type(), "Source");
      if (auto mold{characteristics::TypeAndShape::Characterize(
              arguments[1], foldingContext)}) {
        CheckTransferOperandType(context, mold->type(), "Mold");
        if (mold->Rank() > 0 &&
            evaluate::ToInt64(
                evaluate::Fold(foldingContext,
                    mold->MeasureElementSizeInBytes(foldingContext, false)))
                    .value_or(1) == 0) {
          if (auto sourceSize{evaluate::ToInt64(evaluate::Fold(foldingContext,
                  source->MeasureSizeInBytes(foldingContext)))}) {
            if (*sourceSize > 0) {
              messages.Say(
                  "Element size of MOLD= array may not be zero when SOURCE= is not empty"_err_en_US);
            }
          } else if (context.ShouldWarn(common::UsageWarning::VoidMold)) {
            messages.Say(common::UsageWarning::VoidMold,
                "Element size of MOLD= array may not be zero unless SOURCE= is empty"_warn_en_US);
          }
        }
      }
    }
    if (arguments.size() > 2) { // SIZE=
      if (const Symbol *
          whole{UnwrapWholeSymbolOrComponentDataRef(arguments[2])}) {
        if (IsOptional(*whole)) {
          messages.Say(
              "SIZE= argument may not be the optional dummy argument '%s'"_err_en_US,
              whole->name());
        } else if (context.ShouldWarn(
                       common::UsageWarning::TransferSizePresence) &&
            IsAllocatableOrObjectPointer(whole)) {
          messages.Say(common::UsageWarning::TransferSizePresence,
              "SIZE= argument that is allocatable or pointer must be present at execution; parenthesize to silence this warning"_warn_en_US);
        }
      }
    }
  }
}

static void CheckSpecificIntrinsic(const characteristics::Procedure &proc,
    evaluate::ActualArguments &arguments, SemanticsContext &context,
    const Scope *scope, const evaluate::SpecificIntrinsic &intrinsic) {
  if (intrinsic.name == "associated") {
    CheckAssociated(arguments, context, scope);
  } else if (intrinsic.name == "co_reduce") {
    CheckCoReduce(arguments, context.foldingContext());
  } else if (intrinsic.name == "event_query") {
    CheckEvent_Query(arguments, context.foldingContext());
  } else if (intrinsic.name == "image_index") {
    CheckImage_Index(arguments, context.foldingContext().messages());
  } else if (intrinsic.name == "max" || intrinsic.name == "min") {
    CheckMaxMin(proc, arguments, context.foldingContext().messages());
  } else if (intrinsic.name == "move_alloc") {
    CheckMove_Alloc(arguments, context.foldingContext().messages());
  } else if (intrinsic.name == "present") {
    CheckPresent(arguments, context.foldingContext().messages());
  } else if (intrinsic.name == "reduce") {
    CheckReduce(arguments, context.foldingContext());
  } else if (intrinsic.name == "transfer") {
    CheckTransfer(arguments, context, scope);
  } else if (intrinsic.name == "free") {
    CheckFree(arguments, context.foldingContext().messages());
  }
}

static parser::Messages CheckExplicitInterface(
    const characteristics::Procedure &proc, evaluate::ActualArguments &actuals,
    SemanticsContext &context, const Scope *scope,
    const evaluate::SpecificIntrinsic *intrinsic,
    bool allowActualArgumentConversions, bool extentErrors,
    bool ignoreImplicitVsExplicit) {
  evaluate::FoldingContext &foldingContext{context.foldingContext()};
  parser::ContextualMessages &messages{foldingContext.messages()};
  parser::Messages buffer;
  auto restorer{messages.SetMessages(buffer)};
  RearrangeArguments(proc, actuals, messages);
  if (!buffer.empty()) {
    return buffer;
  }
  int index{0};
  for (auto &actual : actuals) {
    const auto &dummy{proc.dummyArguments.at(index++)};
    if (actual) {
      CheckExplicitInterfaceArg(*actual, dummy, proc, context, scope, intrinsic,
          allowActualArgumentConversions, extentErrors,
          ignoreImplicitVsExplicit);
    } else if (!dummy.IsOptional()) {
      if (dummy.name.empty()) {
        messages.Say(
            "Dummy argument #%d is not OPTIONAL and is not associated with "
            "an actual argument in this procedure reference"_err_en_US,
            index);
      } else {
        messages.Say("Dummy argument '%s=' (#%d) is not OPTIONAL and is not "
                     "associated with an actual argument in this procedure "
                     "reference"_err_en_US,
            dummy.name, index);
      }
    }
  }
  if (proc.IsElemental() && !buffer.AnyFatalError()) {
    CheckElementalConformance(messages, proc, actuals, foldingContext);
  }
  if (intrinsic) {
    CheckSpecificIntrinsic(proc, actuals, context, scope, *intrinsic);
  }
  return buffer;
}

bool CheckInterfaceForGeneric(const characteristics::Procedure &proc,
    evaluate::ActualArguments &actuals, SemanticsContext &context,
    bool allowActualArgumentConversions) {
  return proc.HasExplicitInterface() &&
      !CheckExplicitInterface(proc, actuals, context, nullptr, nullptr,
          allowActualArgumentConversions, /*extentErrors=*/false,
          /*ignoreImplicitVsExplicit=*/false)
           .AnyFatalError();
}

bool CheckArgumentIsConstantExprInRange(
    const evaluate::ActualArguments &actuals, int index, int lowerBound,
    int upperBound, parser::ContextualMessages &messages) {
  CHECK(index >= 0 && static_cast<unsigned>(index) < actuals.size());

  const std::optional<evaluate::ActualArgument> &argOptional{actuals[index]};
  if (!argOptional) {
    DIE("Actual argument should have value");
    return false;
  }

  const evaluate::ActualArgument &arg{argOptional.value()};
  const evaluate::Expr<evaluate::SomeType> *argExpr{arg.UnwrapExpr()};
  CHECK(argExpr != nullptr);

  if (!IsConstantExpr(*argExpr)) {
    messages.Say("Actual argument #%d must be a constant expression"_err_en_US,
        index + 1);
    return false;
  }

  // This does not imply that the kind of the argument is 8. The kind
  // for the intrinsic's argument should have been check prior. This is just
  // a conversion so that we can read the constant value.
  auto scalarValue{evaluate::ToInt64(argExpr)};
  CHECK(scalarValue.has_value());

  if (*scalarValue < lowerBound || *scalarValue > upperBound) {
    messages.Say(
        "Argument #%d must be a constant expression in range %d to %d"_err_en_US,
        index + 1, lowerBound, upperBound);
    return false;
  }
  return true;
}

bool CheckPPCIntrinsic(const Symbol &generic, const Symbol &specific,
    const evaluate::ActualArguments &actuals,
    evaluate::FoldingContext &context) {
  parser::ContextualMessages &messages{context.messages()};

  if (specific.name() == "__ppc_mtfsf") {
    return CheckArgumentIsConstantExprInRange(actuals, 0, 0, 7, messages);
  }
  if (specific.name() == "__ppc_mtfsfi") {
    return CheckArgumentIsConstantExprInRange(actuals, 0, 0, 7, messages) &&
        CheckArgumentIsConstantExprInRange(actuals, 1, 0, 15, messages);
  }
  if (specific.name().ToString().compare(0, 14, "__ppc_vec_sld_") == 0) {
    return CheckArgumentIsConstantExprInRange(actuals, 2, 0, 15, messages);
  }
  if (specific.name().ToString().compare(0, 15, "__ppc_vec_sldw_") == 0) {
    return CheckArgumentIsConstantExprInRange(actuals, 2, 0, 3, messages);
  }
  if (specific.name().ToString().compare(0, 14, "__ppc_vec_ctf_") == 0) {
    return CheckArgumentIsConstantExprInRange(actuals, 1, 0, 31, messages);
  }
  if (specific.name().ToString().compare(0, 16, "__ppc_vec_permi_") == 0) {
    return CheckArgumentIsConstantExprInRange(actuals, 2, 0, 3, messages);
  }
  if (specific.name().ToString().compare(0, 21, "__ppc_vec_splat_s32__") == 0) {
    return CheckArgumentIsConstantExprInRange(actuals, 0, -16, 15, messages);
  }
  if (specific.name().ToString().compare(0, 16, "__ppc_vec_splat_") == 0) {
    // The value of arg2 in vec_splat must be a constant expression that is
    // greater than or equal to 0, and less than the number of elements in arg1.
    auto *expr{actuals[0].value().UnwrapExpr()};
    auto type{characteristics::TypeAndShape::Characterize(*expr, context)};
    assert(type && "unknown type");
    const auto *derived{evaluate::GetDerivedTypeSpec(type.value().type())};
    if (derived && derived->IsVectorType()) {
      for (const auto &pair : derived->parameters()) {
        if (pair.first == "element_kind") {
          auto vecElemKind{Fortran::evaluate::ToInt64(pair.second.GetExplicit())
                               .value_or(0)};
          auto numElem{vecElemKind == 0 ? 0 : (16 / vecElemKind)};
          return CheckArgumentIsConstantExprInRange(
              actuals, 1, 0, numElem - 1, messages);
        }
      }
    } else
      assert(false && "vector type is expected");
  }
  return false;
}

bool CheckWindowsIntrinsic(
    const Symbol &intrinsic, evaluate::FoldingContext &foldingContext) {
  parser::ContextualMessages &messages{foldingContext.messages()};
  // TODO: there are other intrinsics that are unsupported on Windows that
  // should be added here.
  if (intrinsic.name() == "getuid") {
    messages.Say(
        "User IDs do not exist on Windows. This function will always return 1"_warn_en_US);
  }
  if (intrinsic.name() == "getgid") {
    messages.Say(
        "Group IDs do not exist on Windows. This function will always return 1"_warn_en_US);
  }
  return true;
}

bool CheckArguments(const characteristics::Procedure &proc,
    evaluate::ActualArguments &actuals, SemanticsContext &context,
    const Scope &scope, bool treatingExternalAsImplicit,
    bool ignoreImplicitVsExplicit,
    const evaluate::SpecificIntrinsic *intrinsic) {
  bool explicitInterface{proc.HasExplicitInterface()};
  evaluate::FoldingContext foldingContext{context.foldingContext()};
  parser::ContextualMessages &messages{foldingContext.messages()};
  bool allowArgumentConversions{true};
  if (!explicitInterface || treatingExternalAsImplicit) {
    parser::Messages buffer;
    {
      auto restorer{messages.SetMessages(buffer)};
      for (auto &actual : actuals) {
        if (actual) {
          CheckImplicitInterfaceArg(*actual, messages, context);
        }
      }
    }
    if (!buffer.empty()) {
      if (auto *msgs{messages.messages()}) {
        msgs->Annex(std::move(buffer));
      }
      return false; // don't pile on
    }
    allowArgumentConversions = false;
  }
  if (explicitInterface) {
    auto buffer{CheckExplicitInterface(proc, actuals, context, &scope,
        intrinsic, allowArgumentConversions,
        /*extentErrors=*/true, ignoreImplicitVsExplicit)};
    if (!buffer.empty()) {
      if (treatingExternalAsImplicit) {
        if (context.ShouldWarn(
                common::UsageWarning::KnownBadImplicitInterface)) {
          if (auto *msg{messages.Say(
                  common::UsageWarning::KnownBadImplicitInterface,
                  "If the procedure's interface were explicit, this reference would be in error"_warn_en_US)}) {
            buffer.AttachTo(*msg, parser::Severity::Because);
          }
        } else {
          buffer.clear();
        }
      }
      if (auto *msgs{messages.messages()}) {
        msgs->Annex(std::move(buffer));
      }
      return false;
    }
  }
  return true;
}
} // namespace Fortran::semantics
