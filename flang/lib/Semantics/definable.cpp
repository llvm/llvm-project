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
  message.set_severity(parser::Severity::Because);
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

// When a DataRef contains pointers, gets the rightmost one (unless it is
// the entity being defined, in which case the last pointer above it);
// otherwise, returns the leftmost symbol.  The resulting symbol is the
// relevant base object for definabiliy checking.  Examples:
//   ptr1%ptr2        => ...     -> ptr1
//   nonptr%ptr       => ...     -> nonptr
//   nonptr%ptr       =  ...     -> ptr
//   ptr1%ptr2        =  ...     -> ptr2
//   ptr1%ptr2%nonptr =  ...     -> ptr2
//   nonptr1%nonptr2  =  ...     -> nonptr1
static const Symbol &GetRelevantSymbol(const evaluate::DataRef &dataRef,
    bool isPointerDefinition, bool acceptAllocatable) {
  if (isPointerDefinition) {
    if (const auto *component{std::get_if<evaluate::Component>(&dataRef.u)}) {
      if (IsPointer(component->GetLastSymbol()) ||
          (acceptAllocatable && IsAllocatable(component->GetLastSymbol()))) {
        return GetRelevantSymbol(component->base(), false, false);
      }
    }
  }
  if (const Symbol * lastPointer{GetLastPointerSymbol(dataRef)}) {
    return *lastPointer;
  } else {
    return dataRef.GetFirstSymbol();
  }
}

// Check the leftmost (or only) symbol from a data-ref or expression.
static std::optional<parser::Message> WhyNotDefinableBase(parser::CharBlock at,
    const Scope &scope, DefinabilityFlags flags, const Symbol &original) {
  const Symbol &ultimate{original.GetUltimate()};
  bool isPointerDefinition{flags.test(DefinabilityFlag::PointerDefinition)};
  bool acceptAllocatable{flags.test(DefinabilityFlag::AcceptAllocatable)};
  bool isTargetDefinition{!isPointerDefinition && IsPointer(ultimate)};
  if (const auto *association{ultimate.detailsIf<AssocEntityDetails>()}) {
    if (association->rank().has_value()) {
      return std::nullopt; // SELECT RANK always modifiable variable
    } else if (!IsVariable(association->expr())) {
      return BlameSymbol(at,
          "'%s' is construct associated with an expression"_en_US, original);
    } else if (evaluate::HasVectorSubscript(association->expr().value())) {
      return BlameSymbol(at,
          "Construct association '%s' has a vector subscript"_en_US, original);
    } else if (auto dataRef{evaluate::ExtractDataRef(
                   *association->expr(), true, true)}) {
      return WhyNotDefinableBase(at, scope, flags,
          GetRelevantSymbol(*dataRef, isPointerDefinition, acceptAllocatable));
    }
  }
  if (isTargetDefinition) {
  } else if (!isPointerDefinition && !IsVariableName(ultimate)) {
    return BlameSymbol(at, "'%s' is not a variable"_en_US, original);
  } else if (IsProtected(ultimate) && IsUseAssociated(original, scope)) {
    return BlameSymbol(at, "'%s' is protected in this scope"_en_US, original);
  } else if (IsIntentIn(ultimate)) {
    return BlameSymbol(
        at, "'%s' is an INTENT(IN) dummy argument"_en_US, original);
  }
  if (const Scope * pure{FindPureProcedureContaining(scope)}) {
    // Additional checking for pure subprograms.
    if (!isTargetDefinition) {
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
  return std::nullopt;
}

static std::optional<parser::Message> WhyNotDefinableLast(parser::CharBlock at,
    const Scope &scope, DefinabilityFlags flags, const Symbol &original) {
  const Symbol &ultimate{original.GetUltimate()};
  if (flags.test(DefinabilityFlag::PointerDefinition)) {
    if (flags.test(DefinabilityFlag::AcceptAllocatable)) {
      if (!IsAllocatableOrPointer(ultimate)) {
        return BlameSymbol(
            at, "'%s' is neither a pointer nor an allocatable"_en_US, original);
      }
    } else if (!IsPointer(ultimate)) {
      return BlameSymbol(at, "'%s' is not a pointer"_en_US, original);
    }
    return std::nullopt; // pointer assignment - skip following checks
  }
  if (IsOrContainsEventOrLockComponent(ultimate)) {
    return BlameSymbol(at,
        "'%s' is an entity with either an EVENT_TYPE or LOCK_TYPE"_en_US,
        original);
  }
  if (!flags.test(DefinabilityFlag::PolymorphicOkInPure) &&
      FindPureProcedureContaining(scope)) {
    if (auto dyType{evaluate::DynamicType::From(ultimate)}) {
      if (dyType->IsPolymorphic()) { // C1596
        return BlameSymbol(at,
            "'%s' is polymorphic in a pure subprogram"_because_en_US, original);
      }
      if (const DerivedTypeSpec * derived{GetDerivedTypeSpec(dyType)}) {
        if (auto bad{FindPolymorphicAllocatableNonCoarrayUltimateComponent(
                *derived)}) {
          return BlameSymbol(at,
              "'%s' has polymorphic non-coarray component '%s' in a pure subprogram"_because_en_US,
              original, bad.BuildResultDesignatorName());
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
  const Symbol &base{GetRelevantSymbol(dataRef,
      flags.test(DefinabilityFlag::PointerDefinition),
      flags.test(DefinabilityFlag::AcceptAllocatable))};
  if (auto whyNot{WhyNotDefinableBase(at, scope, flags, base)}) {
    return whyNot;
  } else {
    return WhyNotDefinableLast(at, scope, flags, dataRef.GetLastSymbol());
  }
}

// Checks a NOPASS procedure pointer component
static std::optional<parser::Message> WhyNotDefinable(parser::CharBlock at,
    const Scope &scope, DefinabilityFlags flags,
    const evaluate::Component &component) {
  const evaluate::DataRef &dataRef{component.base()};
  const Symbol &base{GetRelevantSymbol(dataRef, false, false)};
  DefinabilityFlags baseFlags{flags};
  baseFlags.reset(DefinabilityFlag::PointerDefinition);
  return WhyNotDefinableBase(at, scope, baseFlags, base);
}

std::optional<parser::Message> WhyNotDefinable(parser::CharBlock at,
    const Scope &scope, DefinabilityFlags flags, const Symbol &original) {
  if (auto base{WhyNotDefinableBase(at, scope, flags, original)}) {
    return base;
  }
  return WhyNotDefinableLast(at, scope, flags, original);
}

std::optional<parser::Message> WhyNotDefinable(parser::CharBlock at,
    const Scope &scope, DefinabilityFlags flags,
    const evaluate::Expr<evaluate::SomeType> &expr) {
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
              for (const auto &[_, ref] :
                  spec->typeSymbol().get<DerivedTypeDetails>().finals()) {
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
                return parser::Message{at,
                    "Variable '%s' has a vector subscript and cannot be finalized by non-elemental subroutine '%s'"_because_en_US,
                    expr.AsFortran(), anyRankMatch->name()};
              }
              const auto *parent{FindParentTypeSpec(*spec)};
              spec = parent ? parent->AsDerived() : nullptr;
            }
          }
        }
      } else {
        return parser::Message{at,
            "Variable '%s' has a vector subscript"_because_en_US,
            expr.AsFortran()};
      }
    }
    if (FindPureProcedureContaining(scope) &&
        evaluate::ExtractCoarrayRef(expr)) {
      return parser::Message(at,
          "A pure subprogram may not define the coindexed object '%s'"_because_en_US,
          expr.AsFortran());
    }
    return WhyNotDefinable(at, scope, flags, *dataRef);
  } else if (evaluate::IsNullPointer(expr)) {
    return parser::Message{
        at, "'%s' is a null pointer"_because_en_US, expr.AsFortran()};
  } else if (flags.test(DefinabilityFlag::PointerDefinition)) {
    if (const auto *procDesignator{
            std::get_if<evaluate::ProcedureDesignator>(&expr.u)}) {
      // Defining a procedure pointer
      if (const Symbol * procSym{procDesignator->GetSymbol()}) {
        if (evaluate::ExtractCoarrayRef(expr)) { // C1027
          return BlameSymbol(at,
              "Procedure pointer '%s' may not be a coindexed object"_because_en_US,
              *procSym, expr.AsFortran());
        }
        if (const auto *component{procDesignator->GetComponent()}) {
          return WhyNotDefinable(at, scope, flags, *component);
        } else {
          return WhyNotDefinable(at, scope, flags, *procSym);
        }
      }
    }
    return parser::Message{
        at, "'%s' is not a definable pointer"_because_en_US, expr.AsFortran()};
  } else if (!evaluate::IsVariable(expr)) {
    return parser::Message{at,
        "'%s' is not a variable or pointer"_because_en_US, expr.AsFortran()};
  } else {
    return std::nullopt;
  }
}

} // namespace Fortran::semantics
