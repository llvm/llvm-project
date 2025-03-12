//===-- lib/Semantics/definable.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "definable.h"
#include "flang/Evaluate/tools.h"
#include "flang/Semantics/tools.h"

using namespace Fortran::parser::literals;

namespace Fortran::semantics {

template <typename... A>
static parser::Message BlameSymbol(parser::CharBlock at,
    const parser::MessageFixedText &text, const Symbol &original, A &&...x) {
  parser::Message message{at, text, original.name(), std::forward<A>(x)...};
  message.set_severity(parser::Severity::Error);
  evaluate::AttachDeclaration(message, original);
  return message;
}

static bool IsPointerDummyOfPureFunction(const Symbol &x) {
  return IsPointerDummy(x) && FindPureProcedureContaining(x.owner()) &&
      x.owner().symbol() && IsFunction(*x.owner().symbol());
}

// See C1594, first paragraph.  These conditions enable checks on both
// left-hand and right-hand sides in various circumstances.
const char *WhyBaseObjectIsSuspicious(const Symbol &x, const Scope &scope) {
  if (IsHostAssociatedIntoSubprogram(x, scope)) {
    return "host-associated";
  } else if (IsUseAssociated(x, scope)) {
    return "USE-associated";
  } else if (IsPointerDummyOfPureFunction(x)) {
    return "a POINTER dummy argument of a pure function";
  } else if (IsIntentIn(x)) {
    return "an INTENT(IN) dummy argument";
  } else if (FindCommonBlockContaining(x)) {
    return "in a COMMON block";
  } else {
    return nullptr;
  }
}

// Checks C1594(1,2); false if check fails
static std::optional<parser::Message> CheckDefinabilityInPureScope(
    SourceName at, const Symbol &original, const Symbol &ultimate,
    const Scope &context, const Scope &pure) {
  if (pure.symbol()) {
    if (const char *why{WhyBaseObjectIsSuspicious(ultimate, context)}) {
      return BlameSymbol(at,
          "'%s' may not be defined in pure subprogram '%s' because it is %s"_en_US,
          original, pure.symbol()->name(), why);
    }
  }
  return std::nullopt;
}

// True when the object being defined is not a subobject of the base
// object, e.g. X%PTR = 1., X%PTR%PTR2 => T (but not X%PTR => T).
// F'2023 9.4.2p5
static bool DefinesComponentPointerTarget(
    const evaluate::DataRef &dataRef, DefinabilityFlags flags) {
  if (const evaluate::Component *
      component{common::visit(
          common::visitors{
              [](const SymbolRef &) -> const evaluate::Component * {
                return nullptr;
              },
              [](const evaluate::Component &component) { return &component; },
              [](const evaluate::ArrayRef &aRef) {
                return aRef.base().UnwrapComponent();
              },
              [](const evaluate::CoarrayRef &aRef)
                  -> const evaluate::Component * { return nullptr; },
          },
          dataRef.u)}) {
    const Symbol &compSym{component->GetLastSymbol()};
    if (IsPointer(compSym) ||
        (flags.test(DefinabilityFlag::AcceptAllocatable) &&
            IsAllocatable(compSym))) {
      if (!flags.test(DefinabilityFlag::PointerDefinition)) {
        return true;
      }
    }
    flags.reset(DefinabilityFlag::PointerDefinition);
    return DefinesComponentPointerTarget(component->base(), flags);
  } else {
    return false;
  }
}

// Check the leftmost (or only) symbol from a data-ref or expression.
static std::optional<parser::Message> WhyNotDefinableBase(parser::CharBlock at,
    const Scope &scope, DefinabilityFlags flags, const Symbol &original,
    bool isWholeSymbol, bool isComponentPointerTarget) {
  const Symbol &ultimate{original.GetUltimate()};
  bool isPointerDefinition{flags.test(DefinabilityFlag::PointerDefinition)};
  bool acceptAllocatable{flags.test(DefinabilityFlag::AcceptAllocatable)};
  bool isTargetDefinition{!isPointerDefinition && IsPointer(ultimate)};
  if (const auto *association{ultimate.detailsIf<AssocEntityDetails>()}) {
    if (!IsVariable(association->expr())) {
      return BlameSymbol(at,
          "'%s' is construct associated with an expression"_en_US, original);
    } else if (evaluate::HasVectorSubscript(association->expr().value())) {
      return BlameSymbol(at,
          "Construct association '%s' has a vector subscript"_en_US, original);
    } else if (auto dataRef{evaluate::ExtractDataRef(
                   *association->expr(), true, true)}) {
      return WhyNotDefinableBase(at, scope, flags, dataRef->GetFirstSymbol(),
          isWholeSymbol &&
              std::holds_alternative<evaluate::SymbolRef>(dataRef->u),
          isComponentPointerTarget ||
              DefinesComponentPointerTarget(*dataRef, flags));
    }
  }
  if (isTargetDefinition || isComponentPointerTarget) {
  } else if (!isPointerDefinition && !IsVariableName(ultimate)) {
    return BlameSymbol(at, "'%s' is not a variable"_en_US, original);
  } else if (IsProtected(ultimate) && IsUseAssociated(original, scope)) {
    return BlameSymbol(at, "'%s' is protected in this scope"_en_US, original);
  } else if (IsIntentIn(ultimate) &&
      (!IsPointer(ultimate) || (isWholeSymbol && isPointerDefinition))) {
    return BlameSymbol(
        at, "'%s' is an INTENT(IN) dummy argument"_en_US, original);
  } else if (acceptAllocatable && IsAllocatable(ultimate) &&
      !flags.test(DefinabilityFlag::SourcedAllocation)) {
    // allocating a function result doesn't count as a def'n
    // unless there's SOURCE=
  } else if (!flags.test(DefinabilityFlag::DoNotNoteDefinition)) {
    scope.context().NoteDefinedSymbol(ultimate);
  }
  if (const Scope * pure{FindPureProcedureContaining(scope)}) {
    // Additional checking for pure subprograms.
    if (!isTargetDefinition || isComponentPointerTarget) {
      if (auto msg{CheckDefinabilityInPureScope(
              at, original, ultimate, scope, *pure)}) {
        return msg;
      }
    }
    if (const Symbol *
        visible{FindExternallyVisibleObject(
            ultimate, *pure, isPointerDefinition)}) {
      return BlameSymbol(at,
          "'%s' is externally visible via '%s' and not definable in a pure subprogram"_en_US,
          original, visible->name());
    }
  }
  if (const Scope * deviceContext{FindCUDADeviceContext(&scope)}) {
    bool isOwnedByDeviceCode{deviceContext->Contains(ultimate.owner())};
    if (isPointerDefinition && !acceptAllocatable) {
      return BlameSymbol(at,
          "'%s' is a pointer and may not be associated in a device subprogram"_err_en_US,
          original);
    } else if (auto cudaDataAttr{GetCUDADataAttr(&ultimate)}) {
      if (*cudaDataAttr == common::CUDADataAttr::Constant) {
        return BlameSymbol(at,
            "'%s' has ATTRIBUTES(CONSTANT) and is not definable in a device subprogram"_err_en_US,
            original);
      } else if (acceptAllocatable && !isOwnedByDeviceCode) {
        return BlameSymbol(at,
            "'%s' is a host-associated allocatable and is not definable in a device subprogram"_err_en_US,
            original);
      } else if (*cudaDataAttr != common::CUDADataAttr::Device &&
          *cudaDataAttr != common::CUDADataAttr::Managed &&
          *cudaDataAttr != common::CUDADataAttr::Shared) {
        return BlameSymbol(at,
            "'%s' is not device or managed or shared data and is not definable in a device subprogram"_err_en_US,
            original);
      }
    } else if (!isOwnedByDeviceCode) {
      return BlameSymbol(at,
          "'%s' is a host variable and is not definable in a device subprogram"_err_en_US,
          original);
    }
  }
  return std::nullopt;
}

static std::optional<parser::Message> WhyNotDefinableLast(parser::CharBlock at,
    const Scope &scope, DefinabilityFlags flags, const Symbol &original) {
  const Symbol &ultimate{original.GetUltimate()};
  if (const auto *association{ultimate.detailsIf<AssocEntityDetails>()};
      association &&
      (association->rank().has_value() ||
          !flags.test(DefinabilityFlag::PointerDefinition))) {
    if (auto dataRef{
            evaluate::ExtractDataRef(*association->expr(), true, true)}) {
      return WhyNotDefinableLast(at, scope, flags, dataRef->GetLastSymbol());
    }
  }
  if (flags.test(DefinabilityFlag::PointerDefinition)) {
    if (flags.test(DefinabilityFlag::AcceptAllocatable)) {
      if (!IsAllocatableOrObjectPointer(&ultimate)) {
        return BlameSymbol(
            at, "'%s' is neither a pointer nor an allocatable"_en_US, original);
      }
    } else if (!IsPointer(ultimate)) {
      return BlameSymbol(at, "'%s' is not a pointer"_en_US, original);
    }
    return std::nullopt; // pointer assignment - skip following checks
  }
  if (!flags.test(DefinabilityFlag::AllowEventLockOrNotifyType) &&
      IsOrContainsEventOrLockComponent(ultimate)) {
    return BlameSymbol(at,
        "'%s' is an entity with either an EVENT_TYPE or LOCK_TYPE"_en_US,
        original);
  }
  if (FindPureProcedureContaining(scope)) {
    if (auto dyType{evaluate::DynamicType::From(ultimate)}) {
      if (!flags.test(DefinabilityFlag::PolymorphicOkInPure)) {
        if (dyType->IsPolymorphic()) { // C1596
          return BlameSymbol(
              at, "'%s' is polymorphic in a pure subprogram"_en_US, original);
        }
      }
      if (const Symbol * impure{HasImpureFinal(ultimate)}) {
        return BlameSymbol(at, "'%s' has an impure FINAL procedure '%s'"_en_US,
            original, impure->name());
      }
      if (const DerivedTypeSpec * derived{GetDerivedTypeSpec(dyType)}) {
        if (!flags.test(DefinabilityFlag::PolymorphicOkInPure)) {
          if (auto bad{
                  FindPolymorphicAllocatablePotentialComponent(*derived)}) {
            return BlameSymbol(at,
                "'%s' has polymorphic component '%s' in a pure subprogram"_en_US,
                original, bad.BuildResultDesignatorName());
          }
        }
      }
    }
  }
  return std::nullopt;
}

// Checks a data-ref
static std::optional<parser::Message> WhyNotDefinable(parser::CharBlock at,
    const Scope &scope, DefinabilityFlags flags,
    const evaluate::DataRef &dataRef) {
  auto whyNotBase{
      WhyNotDefinableBase(at, scope, flags, dataRef.GetFirstSymbol(),
          std::holds_alternative<evaluate::SymbolRef>(dataRef.u),
          DefinesComponentPointerTarget(dataRef, flags))};
  if (!whyNotBase || !whyNotBase->IsFatal()) {
    if (auto whyNotLast{
            WhyNotDefinableLast(at, scope, flags, dataRef.GetLastSymbol())}) {
      if (whyNotLast->IsFatal() || !whyNotBase) {
        return whyNotLast;
      }
    }
  }
  return whyNotBase;
}

std::optional<parser::Message> WhyNotDefinable(parser::CharBlock at,
    const Scope &scope, DefinabilityFlags flags, const Symbol &original) {
  auto whyNotBase{WhyNotDefinableBase(at, scope, flags, original,
      /*isWholeSymbol=*/true, /*isComponentPointerTarget=*/false)};
  if (!whyNotBase || !whyNotBase->IsFatal()) {
    if (auto whyNotLast{WhyNotDefinableLast(at, scope, flags, original)}) {
      if (whyNotLast->IsFatal() || !whyNotBase) {
        return whyNotLast;
      }
    }
  }
  return whyNotBase;
}

class DuplicatedSubscriptFinder
    : public evaluate::AnyTraverse<DuplicatedSubscriptFinder, bool> {
  using Base = evaluate::AnyTraverse<DuplicatedSubscriptFinder, bool>;

public:
  explicit DuplicatedSubscriptFinder(evaluate::FoldingContext &foldingContext)
      : Base{*this}, foldingContext_{foldingContext} {}
  using Base::operator();
  bool operator()(const evaluate::ActualArgument &) {
    return false; // don't descend into argument expressions
  }
  bool operator()(const evaluate::ArrayRef &aRef) {
    bool anyVector{false};
    for (const auto &ss : aRef.subscript()) {
      if (ss.Rank() > 0) {
        anyVector = true;
        if (const auto *vecExpr{
                std::get_if<evaluate::IndirectSubscriptIntegerExpr>(&ss.u)}) {
          auto folded{evaluate::Fold(foldingContext_,
              evaluate::Expr<evaluate::SubscriptInteger>{vecExpr->value()})};
          if (const auto *con{
                  evaluate::UnwrapConstantValue<evaluate::SubscriptInteger>(
                      folded)}) {
            std::set<std::int64_t> values;
            for (const auto &j : con->values()) {
              if (auto pair{values.emplace(j.ToInt64())}; !pair.second) {
                return true; // duplicate
              }
            }
          }
          return false;
        }
      }
    }
    return anyVector ? false : (*this)(aRef.base());
  }

private:
  evaluate::FoldingContext &foldingContext_;
};

std::optional<parser::Message> WhyNotDefinable(parser::CharBlock at,
    const Scope &scope, DefinabilityFlags flags,
    const evaluate::Expr<evaluate::SomeType> &expr) {
  std::optional<parser::Message> portabilityWarning;
  if (auto dataRef{evaluate::ExtractDataRef(expr, true, true)}) {
    if (evaluate::HasVectorSubscript(expr)) {
      if (flags.test(DefinabilityFlag::VectorSubscriptIsOk)) {
        if (auto type{expr.GetType()}) {
          if (!type->IsUnlimitedPolymorphic() &&
              type->category() == TypeCategory::Derived) {
            // Seek the FINAL subroutine that should but cannot be called
            // for this definition of an array with a vector-valued subscript.
            // If there's an elemental FINAL subroutine, all is well; otherwise,
            // if there is a FINAL subroutine with a matching or assumed rank
            // dummy argument, there's no way to call it.
            int rank{expr.Rank()};
            const DerivedTypeSpec *spec{&type->GetDerivedTypeSpec()};
            while (spec) {
              bool anyElemental{false};
              const Symbol *anyRankMatch{nullptr};
              for (auto ref : FinalsForDerivedTypeInstantiation(*spec)) {
                const Symbol &ultimate{ref->GetUltimate()};
                anyElemental |= ultimate.attrs().test(Attr::ELEMENTAL);
                if (const auto *subp{ultimate.detailsIf<SubprogramDetails>()}) {
                  if (!subp->dummyArgs().empty()) {
                    if (const Symbol * arg{subp->dummyArgs()[0]}) {
                      const auto *object{arg->detailsIf<ObjectEntityDetails>()};
                      if (arg->Rank() == rank ||
                          (object && object->IsAssumedRank())) {
                        anyRankMatch = &*ref;
                      }
                    }
                  }
                }
              }
              if (anyRankMatch && !anyElemental) {
                if (!portabilityWarning &&
                    scope.context().languageFeatures().ShouldWarn(
                        common::UsageWarning::VectorSubscriptFinalization)) {
                  portabilityWarning = parser::Message{
                      common::UsageWarning::VectorSubscriptFinalization, at,
                      "Variable '%s' has a vector subscript and will be finalized by non-elemental subroutine '%s'"_port_en_US,
                      expr.AsFortran(), anyRankMatch->name()};
                }
                break;
              }
              const auto *parent{FindParentTypeSpec(*spec)};
              spec = parent ? parent->AsDerived() : nullptr;
            }
          }
        }
        if (!flags.test(DefinabilityFlag::DuplicatesAreOk) &&
            DuplicatedSubscriptFinder{scope.context().foldingContext()}(expr)) {
          return parser::Message{at,
              "Variable has a vector subscript with a duplicated element"_err_en_US};
        }
      } else {
        return parser::Message{at,
            "Variable '%s' has a vector subscript"_err_en_US, expr.AsFortran()};
      }
    }
    if (FindPureProcedureContaining(scope) &&
        evaluate::ExtractCoarrayRef(expr)) {
      return parser::Message(at,
          "A pure subprogram may not define the coindexed object '%s'"_err_en_US,
          expr.AsFortran());
    }
    if (auto whyNotDataRef{WhyNotDefinable(at, scope, flags, *dataRef)}) {
      return whyNotDataRef;
    }
  } else if (evaluate::IsNullPointer(expr)) {
    return parser::Message{
        at, "'%s' is a null pointer"_err_en_US, expr.AsFortran()};
  } else if (flags.test(DefinabilityFlag::PointerDefinition)) {
    if (const auto *procDesignator{
            std::get_if<evaluate::ProcedureDesignator>(&expr.u)}) {
      // Defining a procedure pointer
      if (const Symbol * procSym{procDesignator->GetSymbol()}) {
        if (evaluate::ExtractCoarrayRef(expr)) { // C1027
          return BlameSymbol(at,
              "Procedure pointer '%s' may not be a coindexed object"_err_en_US,
              *procSym, expr.AsFortran());
        }
        if (const auto *component{procDesignator->GetComponent()}) {
          flags.reset(DefinabilityFlag::PointerDefinition);
          return WhyNotDefinableBase(at, scope, flags,
              component->base().GetFirstSymbol(), false,
              DefinesComponentPointerTarget(component->base(), flags));
        } else {
          return WhyNotDefinable(at, scope, flags, *procSym);
        }
      }
    }
    return parser::Message{
        at, "'%s' is not a definable pointer"_err_en_US, expr.AsFortran()};
  } else if (!evaluate::IsVariable(expr)) {
    return parser::Message{
        at, "'%s' is not a variable or pointer"_err_en_US, expr.AsFortran()};
  }
  return portabilityWarning;
}

} // namespace Fortran::semantics
