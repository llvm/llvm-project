//===-- lib/Semantics/check-call.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-call.h"
#include "pointer-assignment.h"
#include "flang/Evaluate/characteristics.h"
#include "flang/Evaluate/check-expression.h"
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

static void CheckImplicitInterfaceArg(
    evaluate::ActualArgument &arg, parser::ContextualMessages &messages) {
  if (auto kw{arg.keyword()}) {
    messages.Say(*kw,
        "Keyword '%s=' may not appear in a reference to a procedure with an implicit interface"_err_en_US,
        *kw);
  }
  if (auto type{arg.GetType()}) {
    if (type->IsAssumedType()) {
      messages.Say(
          "Assumed type argument requires an explicit interface"_err_en_US);
    } else if (type->IsPolymorphic()) {
      messages.Say(
          "Polymorphic argument requires an explicit interface"_err_en_US);
    } else if (const DerivedTypeSpec * derived{GetDerivedTypeSpec(type)}) {
      if (!derived->parameters().empty()) {
        messages.Say(
            "Parameterized derived type argument requires an explicit interface"_err_en_US);
      }
    }
  }
  if (const auto *expr{arg.UnwrapExpr()}) {
    if (auto named{evaluate::ExtractNamedEntity(*expr)}) {
      const Symbol &symbol{named->GetLastSymbol()};
      if (symbol.Corank() > 0) {
        messages.Say(
            "Coarray argument requires an explicit interface"_err_en_US);
      }
      if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
        if (details->IsAssumedRank()) {
          messages.Say(
              "Assumed rank argument requires an explicit interface"_err_en_US);
        }
      }
      if (symbol.attrs().test(Attr::ASYNCHRONOUS)) {
        messages.Say(
            "ASYNCHRONOUS argument requires an explicit interface"_err_en_US);
      }
      if (symbol.attrs().test(Attr::VOLATILE)) {
        messages.Say(
            "VOLATILE argument requires an explicit interface"_err_en_US);
      }
    }
  }
}

// When scalar CHARACTER actual arguments are known to be short,
// we extend them on the right with spaces and a warning.
static void PadShortCharacterActual(evaluate::Expr<evaluate::SomeType> &actual,
    const characteristics::TypeAndShape &dummyType,
    const characteristics::TypeAndShape &actualType,
    parser::ContextualMessages &messages) {
  if (dummyType.type().category() == TypeCategory::Character &&
      actualType.type().category() == TypeCategory::Character &&
      dummyType.type().kind() == actualType.type().kind() &&
      GetRank(actualType.shape()) == 0) {
    if (auto dummyLEN{ToInt64(dummyType.LEN())}) {
      if (auto actualLEN{ToInt64(actualType.LEN())}) {
        if (*actualLEN < *dummyLEN) {
          messages.Say(
              "Actual length '%jd' is less than expected length '%jd'"_en_US,
              *actualLEN, *dummyLEN);
          auto converted{ConvertToType(dummyType.type(), std::move(actual))};
          CHECK(converted);
          actual = std::move(*converted);
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
    parser::ContextualMessages &messages) {
  if (dummyType.type().category() == TypeCategory::Integer &&
      actualType.type().category() == TypeCategory::Integer &&
      dummyType.type().kind() != actualType.type().kind() &&
      GetRank(dummyType.shape()) == 0 && GetRank(actualType.shape()) == 0 &&
      !evaluate::IsVariable(actual)) {
    auto converted{
        evaluate::ConvertToType(dummyType.type(), std::move(actual))};
    CHECK(converted);
    actual = std::move(*converted);
    if (dummyType.type().kind() < actualType.type().kind()) {
      messages.Say(
          "Actual argument scalar expression of type INTEGER(%d) was converted to smaller dummy argument type INTEGER(%d)"_en_US,
          actualType.type().kind(), dummyType.type().kind());
    }
    actualType = dummyType;
  }
}

static bool DefersSameTypeParameters(
    const DerivedTypeSpec &actual, const DerivedTypeSpec &dummy) {
  for (const auto &pair : actual.parameters()) {
    const ParamValue &actualValue{pair.second};
    const ParamValue *dummyValue{dummy.FindParameter(pair.first)};
    if (!dummyValue || (actualValue.isDeferred() != dummyValue->isDeferred())) {
      return false;
    }
  }
  return true;
}

static void CheckExplicitDataArg(const characteristics::DummyDataObject &dummy,
    const std::string &dummyName, evaluate::Expr<evaluate::SomeType> &actual,
    characteristics::TypeAndShape &actualType, bool isElemental,
    bool actualIsArrayElement, evaluate::FoldingContext &context,
    const Scope *scope) {

  // Basic type & rank checking
  parser::ContextualMessages &messages{context.messages()};
  PadShortCharacterActual(actual, dummy.type, actualType, messages);
  ConvertIntegerActual(actual, dummy.type, actualType, messages);
  bool typesCompatible{
      dummy.type.type().IsTypeCompatibleWith(actualType.type())};
  if (typesCompatible) {
    if (isElemental) {
    } else if (dummy.type.attrs().test(
                   characteristics::TypeAndShape::Attr::AssumedRank)) {
    } else if (!dummy.type.attrs().test(
                   characteristics::TypeAndShape::Attr::AssumedShape) &&
        (actualType.Rank() > 0 || actualIsArrayElement)) {
      // Sequence association (15.5.2.11) applies -- rank need not match
      // if the actual argument is an array or array element designator.
    } else {
      CheckConformance(messages, dummy.type.shape(), actualType.shape(),
          "dummy argument", "actual argument");
    }
  } else {
    const auto &len{actualType.LEN()};
    messages.Say(
        "Actual argument type '%s' is not compatible with dummy argument type '%s'"_err_en_US,
        actualType.type().AsFortran(len ? len->AsFortran() : ""),
        dummy.type.type().AsFortran());
  }

  bool actualIsPolymorphic{actualType.type().IsPolymorphic()};
  bool dummyIsPolymorphic{dummy.type.type().IsPolymorphic()};
  bool actualIsCoindexed{ExtractCoarrayRef(actual).has_value()};
  bool actualIsAssumedSize{actualType.attrs().test(
      characteristics::TypeAndShape::Attr::AssumedSize)};
  bool dummyIsAssumedSize{dummy.type.attrs().test(
      characteristics::TypeAndShape::Attr::AssumedSize)};
  bool dummyIsAsynchronous{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Asynchronous)};
  bool dummyIsVolatile{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Volatile)};
  bool dummyIsValue{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Value)};

  if (actualIsPolymorphic && dummyIsPolymorphic &&
      actualIsCoindexed) { // 15.5.2.4(2)
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
  if (const auto *derived{evaluate::GetDerivedTypeSpec(actualType.type())}) {
    if (dummy.type.type().IsAssumedType()) {
      if (!derived->parameters().empty()) { // 15.5.2.4(2)
        messages.Say(
            "Actual argument associated with TYPE(*) %s may not have a parameterized derived type"_err_en_US,
            dummyName);
      }
      if (const Symbol *
          tbp{FindImmediateComponent(*derived, [](const Symbol &symbol) {
            return symbol.has<ProcBindingDetails>();
          })}) { // 15.5.2.4(2)
        evaluate::SayWithDeclaration(messages, *tbp,
            "Actual argument associated with TYPE(*) %s may not have type-bound procedure '%s'"_err_en_US,
            dummyName, tbp->name());
      }
      if (const Symbol *
          finalizer{FindImmediateComponent(*derived, [](const Symbol &symbol) {
            return symbol.has<FinalProcDetails>();
          })}) { // 15.5.2.4(2)
        evaluate::SayWithDeclaration(messages, *finalizer,
            "Actual argument associated with TYPE(*) %s may not have FINAL subroutine '%s'"_err_en_US,
            dummyName, finalizer->name());
      }
    }
    if (actualIsCoindexed) {
      if (dummy.intent != common::Intent::In && !dummyIsValue) {
        if (auto bad{
                FindAllocatableUltimateComponent(*derived)}) { // 15.5.2.4(6)
          evaluate::SayWithDeclaration(messages, *bad,
              "Coindexed actual argument with ALLOCATABLE ultimate component '%s' must be associated with a %s with VALUE or INTENT(IN) attributes"_err_en_US,
              bad.BuildResultDesignatorName(), dummyName);
        }
      }
      if (auto coarrayRef{evaluate::ExtractCoarrayRef(actual)}) { // C1537
        const Symbol &coarray{coarrayRef->GetLastSymbol()};
        if (const DeclTypeSpec * type{coarray.GetType()}) {
          if (const DerivedTypeSpec * derived{type->AsDerived()}) {
            if (auto bad{semantics::FindPointerUltimateComponent(*derived)}) {
              evaluate::SayWithDeclaration(messages, coarray,
                  "Coindexed object '%s' with POINTER ultimate component '%s' cannot be associated with %s"_err_en_US,
                  coarray.name(), bad.BuildResultDesignatorName(), dummyName);
            }
          }
        }
      }
    }
    if (actualIsVolatile != dummyIsVolatile) { // 15.5.2.4(22)
      if (auto bad{semantics::FindCoarrayUltimateComponent(*derived)}) {
        evaluate::SayWithDeclaration(messages, *bad,
            "VOLATILE attribute must match for %s when actual argument has a coarray ultimate component '%s'"_err_en_US,
            dummyName, bad.BuildResultDesignatorName());
      }
    }
  }

  // Rank and shape checks
  const auto *actualLastSymbol{evaluate::GetLastSymbol(actual)};
  if (actualLastSymbol) {
    actualLastSymbol = GetAssociationRoot(*actualLastSymbol);
  }
  const ObjectEntityDetails *actualLastObject{actualLastSymbol
          ? actualLastSymbol->GetUltimate().detailsIf<ObjectEntityDetails>()
          : nullptr};
  int actualRank{evaluate::GetRank(actualType.shape())};
  bool actualIsPointer{(actualLastSymbol && IsPointer(*actualLastSymbol)) ||
      evaluate::IsNullPointer(actual)};
  if (dummy.type.attrs().test(
          characteristics::TypeAndShape::Attr::AssumedShape)) {
    // 15.5.2.4(16)
    if (actualRank == 0) {
      messages.Say(
          "Scalar actual argument may not be associated with assumed-shape %s"_err_en_US,
          dummyName);
    }
    if (actualIsAssumedSize && actualLastSymbol) {
      evaluate::SayWithDeclaration(messages, *actualLastSymbol,
          "Assumed-size array may not be associated with assumed-shape %s"_err_en_US,
          dummyName);
    }
  } else if (actualRank == 0 && dummy.type.Rank() > 0) {
    // Actual is scalar, dummy is an array.  15.5.2.4(14), 15.5.2.11
    if (actualIsCoindexed) {
      messages.Say(
          "Coindexed scalar actual argument must be associated with a scalar %s"_err_en_US,
          dummyName);
    }
    if (actualLastSymbol && actualLastSymbol->Rank() == 0 &&
        !(dummy.type.type().IsAssumedType() && dummyIsAssumedSize)) {
      messages.Say(
          "Whole scalar actual argument may not be associated with a %s array"_err_en_US,
          dummyName);
    }
    if (actualIsPolymorphic) {
      messages.Say(
          "Polymorphic scalar may not be associated with a %s array"_err_en_US,
          dummyName);
    }
    if (actualIsPointer) {
      messages.Say(
          "Scalar POINTER target may not be associated with a %s array"_err_en_US,
          dummyName);
    }
    if (actualLastObject && actualLastObject->IsAssumedShape()) {
      messages.Say(
          "Element of assumed-shape array may not be associated with a %s array"_err_en_US,
          dummyName);
    }
  }
  if (actualLastObject && actualLastObject->IsCoarray() &&
      IsAllocatable(*actualLastSymbol) &&
      dummy.intent == common::Intent::Out) { // C846
    messages.Say(
        "ALLOCATABLE coarray '%s' may not be associated with INTENT(OUT) %s"_err_en_US,
        actualLastSymbol->name(), dummyName);
  }

  // Definability
  const char *reason{nullptr};
  if (dummy.intent == common::Intent::Out) {
    reason = "INTENT(OUT)";
  } else if (dummy.intent == common::Intent::InOut) {
    reason = "INTENT(IN OUT)";
  } else if (dummyIsAsynchronous) {
    reason = "ASYNCHRONOUS";
  } else if (dummyIsVolatile) {
    reason = "VOLATILE";
  }
  if (reason && scope) {
    bool vectorSubscriptIsOk{isElemental || dummyIsValue}; // 15.5.2.4(21)
    if (auto why{WhyNotModifiable(
            messages.at(), actual, *scope, vectorSubscriptIsOk)}) {
      if (auto *msg{messages.Say(
              "Actual argument associated with %s %s must be definable"_err_en_US, // C1158
              reason, dummyName)}) {
        msg->Attach(*why);
      }
    }
  }

  // Cases when temporaries might be needed but must not be permitted.
  bool dummyIsPointer{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Pointer)};
  bool dummyIsContiguous{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Contiguous)};
  bool actualIsContiguous{IsSimplyContiguous(actual, context.intrinsics())};
  bool dummyIsAssumedRank{dummy.type.attrs().test(
      characteristics::TypeAndShape::Attr::AssumedRank)};
  bool dummyIsAssumedShape{dummy.type.attrs().test(
      characteristics::TypeAndShape::Attr::AssumedShape)};
  if ((actualIsAsynchronous || actualIsVolatile) &&
      (dummyIsAsynchronous || dummyIsVolatile) && !dummyIsValue) {
    if (actualIsCoindexed) { // C1538
      messages.Say(
          "Coindexed ASYNCHRONOUS or VOLATILE actual argument may not be associated with %s with ASYNCHRONOUS or VOLATILE attributes unless VALUE"_err_en_US,
          dummyName);
    }
    if (actualRank > 0 && !actualIsContiguous) {
      if (dummyIsContiguous ||
          !(dummyIsAssumedShape || dummyIsAssumedRank ||
              (actualIsPointer && dummyIsPointer))) { // C1539 & C1540
        messages.Say(
            "ASYNCHRONOUS or VOLATILE actual argument that is not simply contiguous may not be associated with a contiguous %s"_err_en_US,
            dummyName);
      }
    }
  }

  // 15.5.2.6 -- dummy is ALLOCATABLE
  bool dummyIsAllocatable{
      dummy.attrs.test(characteristics::DummyDataObject::Attr::Allocatable)};
  bool actualIsAllocatable{
      actualLastSymbol && IsAllocatable(*actualLastSymbol)};
  if (dummyIsAllocatable) {
    if (!actualIsAllocatable) {
      messages.Say(
          "ALLOCATABLE %s must be associated with an ALLOCATABLE actual argument"_err_en_US,
          dummyName);
    }
    if (actualIsAllocatable && actualIsCoindexed &&
        dummy.intent != common::Intent::In) {
      messages.Say(
          "ALLOCATABLE %s must have INTENT(IN) to be associated with a coindexed actual argument"_err_en_US,
          dummyName);
    }
    if (!actualIsCoindexed && actualLastSymbol &&
        actualLastSymbol->Corank() != dummy.type.corank()) {
      messages.Say(
          "ALLOCATABLE %s has corank %d but actual argument has corank %d"_err_en_US,
          dummyName, dummy.type.corank(), actualLastSymbol->Corank());
    }
  }

  // 15.5.2.7 -- dummy is POINTER
  if (dummyIsPointer) {
    if (dummyIsContiguous && !actualIsContiguous) {
      messages.Say(
          "Actual argument associated with CONTIGUOUS POINTER %s must be simply contiguous"_err_en_US,
          dummyName);
    }
    if (!actualIsPointer) {
      if (dummy.intent == common::Intent::In) {
        semantics::CheckPointerAssignment(
            context, parser::CharBlock{}, dummyName, dummy, actual);
      } else {
        messages.Say(
            "Actual argument associated with POINTER %s must also be POINTER unless INTENT(IN)"_err_en_US,
            dummyName);
      }
    }
  }

  // 15.5.2.5 -- actual & dummy are both POINTER or both ALLOCATABLE
  if ((actualIsPointer && dummyIsPointer) ||
      (actualIsAllocatable && dummyIsAllocatable)) {
    bool actualIsUnlimited{actualType.type().IsUnlimitedPolymorphic()};
    bool dummyIsUnlimited{dummy.type.type().IsUnlimitedPolymorphic()};
    if (actualIsUnlimited != dummyIsUnlimited) {
      if (typesCompatible) {
        messages.Say(
            "If a POINTER or ALLOCATABLE dummy or actual argument is unlimited polymorphic, both must be so"_err_en_US);
      }
    } else if (dummyIsPolymorphic != actualIsPolymorphic) {
      if (dummy.intent == common::Intent::In && typesCompatible) {
        // extension: allow with warning, rule is only relevant for definables
        messages.Say(
            "If a POINTER or ALLOCATABLE dummy or actual argument is polymorphic, both should be so"_en_US);
      } else {
        messages.Say(
            "If a POINTER or ALLOCATABLE dummy or actual argument is polymorphic, both must be so"_err_en_US);
      }
    } else if (!actualIsUnlimited && typesCompatible) {
      if (!actualType.type().IsTypeCompatibleWith(dummy.type.type())) {
        if (dummy.intent == common::Intent::In) {
          // extension: allow with warning, rule is only relevant for definables
          messages.Say(
              "POINTER or ALLOCATABLE dummy and actual arguments should have the same declared type"_en_US);
        } else {
          messages.Say(
              "POINTER or ALLOCATABLE dummy and actual arguments must have the same declared type"_err_en_US);
        }
      }
      if (const auto *derived{
              evaluate::GetDerivedTypeSpec(actualType.type())}) {
        if (!DefersSameTypeParameters(
                *derived, *evaluate::GetDerivedTypeSpec(dummy.type.type()))) {
          messages.Say(
              "Dummy and actual arguments must defer the same type parameters when POINTER or ALLOCATABLE"_err_en_US);
        }
      }
    }
  }

  // 15.5.2.8 -- coarray dummy arguments
  if (dummy.type.corank() > 0) {
    if (actualType.corank() == 0) {
      messages.Say(
          "Actual argument associated with coarray %s must be a coarray"_err_en_US,
          dummyName);
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
    if (actualRank == dummy.type.Rank() && !actualIsContiguous) {
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
}

static void CheckProcedureArg(evaluate::ActualArgument &arg,
    const characteristics::DummyProcedure &proc, const std::string &dummyName,
    evaluate::FoldingContext &context) {
  parser::ContextualMessages &messages{context.messages()};
  const characteristics::Procedure &interface{proc.procedure.value()};
  if (const auto *expr{arg.UnwrapExpr()}) {
    bool dummyIsPointer{
        proc.attrs.test(characteristics::DummyProcedure::Attr::Pointer)};
    const auto *argProcDesignator{
        std::get_if<evaluate::ProcedureDesignator>(&expr->u)};
    const auto *argProcSymbol{
        argProcDesignator ? argProcDesignator->GetSymbol() : nullptr};
    if (auto argChars{characteristics::DummyArgument::FromActual(
            "actual argument", *expr, context)}) {
      if (auto *argProc{
              std::get_if<characteristics::DummyProcedure>(&argChars->u)}) {
        characteristics::Procedure &argInterface{argProc->procedure.value()};
        argInterface.attrs.reset(characteristics::Procedure::Attr::NullPointer);
        if (!argProcSymbol || argProcSymbol->attrs().test(Attr::INTRINSIC)) {
          // It's ok to pass ELEMENTAL unrestricted intrinsic functions.
          argInterface.attrs.reset(characteristics::Procedure::Attr::Elemental);
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
          }
        }
        if (!interface.IsPure()) {
          // 15.5.2.9(1): if dummy is not pure, actual need not be.
          argInterface.attrs.reset(characteristics::Procedure::Attr::Pure);
        }
        if (interface.HasExplicitInterface()) {
          if (interface != argInterface) {
            messages.Say(
                "Actual argument procedure has interface incompatible with %s"_err_en_US,
                dummyName);
          }
        } else { // 15.5.2.9(2,3)
          if (interface.IsSubroutine() && argInterface.IsFunction()) {
            messages.Say(
                "Actual argument associated with procedure %s is a function but must be a subroutine"_err_en_US,
                dummyName);
          } else if (interface.IsFunction()) {
            if (argInterface.IsFunction()) {
              if (interface.functionResult != argInterface.functionResult) {
                messages.Say(
                    "Actual argument function associated with procedure %s has incompatible result type"_err_en_US,
                    dummyName);
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
    } else if (!(dummyIsPointer && IsNullPointer(*expr))) {
      messages.Say(
          "Actual argument associated with procedure %s is not a procedure"_err_en_US,
          dummyName);
    }
    if (interface.HasExplicitInterface()) {
      if (dummyIsPointer) {
        // 15.5.2.9(5) -- dummy procedure POINTER
        // Interface compatibility has already been checked above by comparison.
        if (proc.intent != common::Intent::In && !IsVariable(*expr)) {
          messages.Say(
              "Actual argument associated with procedure pointer %s must be a POINTER unless INTENT(IN)"_err_en_US,
              dummyName);
        }
      } else { // 15.5.2.9(4) -- dummy procedure is not POINTER
        if (!argProcDesignator) {
          messages.Say(
              "Actual argument associated with non-POINTER procedure %s must be a procedure (and not a procedure pointer)"_err_en_US,
              dummyName);
        }
      }
    }
  } else {
    messages.Say(
        "Assumed-type argument may not be forwarded as procedure %s"_err_en_US,
        dummyName);
  }
}

static void CheckExplicitInterfaceArg(evaluate::ActualArgument &arg,
    const characteristics::DummyArgument &dummy,
    const characteristics::Procedure &proc, evaluate::FoldingContext &context,
    const Scope *scope) {
  auto &messages{context.messages()};
  std::string dummyName{"dummy argument"};
  if (!dummy.name.empty()) {
    dummyName += " '"s + parser::ToLowerCaseLetters(dummy.name) + "='";
  }
  std::visit(
      common::visitors{
          [&](const characteristics::DummyDataObject &object) {
            if (auto *expr{arg.UnwrapExpr()}) {
              if (auto type{characteristics::TypeAndShape::Characterize(
                      *expr, context)}) {
                arg.set_dummyIntent(object.intent);
                bool isElemental{object.type.Rank() == 0 && proc.IsElemental()};
                CheckExplicitDataArg(object, dummyName, *expr, *type,
                    isElemental, IsArrayElement(*expr), context, scope);
              } else if (object.type.type().IsTypelessIntrinsicArgument() &&
                  std::holds_alternative<evaluate::BOZLiteralConstant>(
                      expr->u)) {
                // ok
              } else {
                messages.Say(
                    "Actual argument is not a variable or typed expression"_err_en_US);
              }
            } else {
              const Symbol &assumed{DEREF(arg.GetAssumedTypeDummy())};
              if (!object.type.type().IsAssumedType()) {
                messages.Say(
                    "Assumed-type '%s' may be associated only with an assumed-type %s"_err_en_US,
                    assumed.name(), dummyName);
              } else if (const auto *details{
                             assumed.detailsIf<ObjectEntityDetails>()}) {
                if (!(details->IsAssumedShape() || details->IsAssumedRank())) {
                  messages.Say( // C711
                      "Assumed-type '%s' must be either assumed shape or assumed rank to be associated with assumed-type %s"_err_en_US,
                      assumed.name(), dummyName);
                }
              }
            }
          },
          [&](const characteristics::DummyProcedure &proc) {
            CheckProcedureArg(arg, proc, dummyName, context);
          },
          [&](const characteristics::AlternateReturn &) {
            // TODO check alternate return
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
  for (auto &x : actuals) {
    if (x && x->keyword()) {
      auto emplaced{
          kwArgs.try_emplace(x->keyword()->ToString(), std::move(*x))};
      if (!emplaced.second) {
        messages.Say(*x->keyword(),
            "Argument keyword '%s=' appears on more than one effective argument in this procedure reference"_err_en_US,
            *x->keyword());
      }
      x.reset();
    }
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

static parser::Messages CheckExplicitInterface(
    const characteristics::Procedure &proc, evaluate::ActualArguments &actuals,
    const evaluate::FoldingContext &context, const Scope *scope) {
  parser::Messages buffer;
  parser::ContextualMessages messages{context.messages().at(), &buffer};
  RearrangeArguments(proc, actuals, messages);
  if (buffer.empty()) {
    int index{0};
    evaluate::FoldingContext localContext{context, messages};
    for (auto &actual : actuals) {
      const auto &dummy{proc.dummyArguments.at(index++)};
      if (actual) {
        CheckExplicitInterfaceArg(*actual, dummy, proc, localContext, scope);
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
  }
  return buffer;
}

parser::Messages CheckExplicitInterface(const characteristics::Procedure &proc,
    evaluate::ActualArguments &actuals, const evaluate::FoldingContext &context,
    const Scope &scope) {
  return CheckExplicitInterface(proc, actuals, context, &scope);
}

bool CheckInterfaceForGeneric(const characteristics::Procedure &proc,
    evaluate::ActualArguments &actuals,
    const evaluate::FoldingContext &context) {
  return CheckExplicitInterface(proc, actuals, context, nullptr).empty();
}

void CheckArguments(const characteristics::Procedure &proc,
    evaluate::ActualArguments &actuals, evaluate::FoldingContext &context,
    const Scope &scope, bool treatingExternalAsImplicit) {
  bool explicitInterface{proc.HasExplicitInterface()};
  if (explicitInterface) {
    auto buffer{CheckExplicitInterface(proc, actuals, context, scope)};
    if (treatingExternalAsImplicit && !buffer.empty()) {
      if (auto *msg{context.messages().Say(
              "Warning: if the procedure's interface were explicit, this reference would be in error:"_en_US)}) {
        buffer.AttachTo(*msg);
      }
    }
    if (auto *msgs{context.messages().messages()}) {
      msgs->Merge(std::move(buffer));
    }
  }
  if (!explicitInterface || treatingExternalAsImplicit) {
    for (auto &actual : actuals) {
      if (actual) {
        CheckImplicitInterfaceArg(*actual, context.messages());
      }
    }
  }
}
} // namespace Fortran::semantics
