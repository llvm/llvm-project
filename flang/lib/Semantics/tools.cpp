//===-- lib/Semantics/tools.cpp -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/tools.h"
#include "flang/Common/Fortran.h"
#include "flang/Common/indirection.h"
#include "flang/Parser/dump-parse-tree.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <set>
#include <variant>

namespace Fortran::semantics {

// Find this or containing scope that matches predicate
static const Scope *FindScopeContaining(
    const Scope &start, std::function<bool(const Scope &)> predicate) {
  for (const Scope *scope{&start};; scope = &scope->parent()) {
    if (predicate(*scope)) {
      return scope;
    }
    if (scope->IsGlobal()) {
      return nullptr;
    }
  }
}

const Scope *FindModuleContaining(const Scope &start) {
  return FindScopeContaining(
      start, [](const Scope &scope) { return scope.IsModule(); });
}

const Scope *FindProgramUnitContaining(const Scope &start) {
  return FindScopeContaining(start, [](const Scope &scope) {
    switch (scope.kind()) {
    case Scope::Kind::Module:
    case Scope::Kind::MainProgram:
    case Scope::Kind::Subprogram:
    case Scope::Kind::BlockData:
      return true;
    default:
      return false;
    }
  });
}

const Scope *FindProgramUnitContaining(const Symbol &symbol) {
  return FindProgramUnitContaining(symbol.owner());
}

const Scope *FindPureProcedureContaining(const Scope &start) {
  // N.B. We only need to examine the innermost containing program unit
  // because an internal subprogram of a pure subprogram must also
  // be pure (C1592).
  if (const Scope * scope{FindProgramUnitContaining(start)}) {
    if (IsPureProcedure(*scope)) {
      return scope;
    }
  }
  return nullptr;
}

Tristate IsDefinedAssignment(
    const std::optional<evaluate::DynamicType> &lhsType, int lhsRank,
    const std::optional<evaluate::DynamicType> &rhsType, int rhsRank) {
  if (!lhsType || !rhsType) {
    return Tristate::No; // error or rhs is untyped
  }
  TypeCategory lhsCat{lhsType->category()};
  TypeCategory rhsCat{rhsType->category()};
  if (rhsRank > 0 && lhsRank != rhsRank) {
    return Tristate::Yes;
  } else if (lhsCat != TypeCategory::Derived) {
    return ToTristate(lhsCat != rhsCat &&
        (!IsNumericTypeCategory(lhsCat) || !IsNumericTypeCategory(rhsCat)));
  } else {
    const auto *lhsDerived{evaluate::GetDerivedTypeSpec(lhsType)};
    const auto *rhsDerived{evaluate::GetDerivedTypeSpec(rhsType)};
    if (lhsDerived && rhsDerived && *lhsDerived == *rhsDerived) {
      return Tristate::Maybe; // TYPE(t) = TYPE(t) can be defined or
                              // intrinsic
    } else {
      return Tristate::Yes;
    }
  }
}

bool IsIntrinsicRelational(common::RelationalOperator opr,
    const evaluate::DynamicType &type0, int rank0,
    const evaluate::DynamicType &type1, int rank1) {
  if (!evaluate::AreConformable(rank0, rank1)) {
    return false;
  } else {
    auto cat0{type0.category()};
    auto cat1{type1.category()};
    if (IsNumericTypeCategory(cat0) && IsNumericTypeCategory(cat1)) {
      // numeric types: EQ/NE always ok, others ok for non-complex
      return opr == common::RelationalOperator::EQ ||
          opr == common::RelationalOperator::NE ||
          (cat0 != TypeCategory::Complex && cat1 != TypeCategory::Complex);
    } else {
      // not both numeric: only Character is ok
      return cat0 == TypeCategory::Character && cat1 == TypeCategory::Character;
    }
  }
}

bool IsIntrinsicNumeric(const evaluate::DynamicType &type0) {
  return IsNumericTypeCategory(type0.category());
}
bool IsIntrinsicNumeric(const evaluate::DynamicType &type0, int rank0,
    const evaluate::DynamicType &type1, int rank1) {
  return evaluate::AreConformable(rank0, rank1) &&
      IsNumericTypeCategory(type0.category()) &&
      IsNumericTypeCategory(type1.category());
}

bool IsIntrinsicLogical(const evaluate::DynamicType &type0) {
  return type0.category() == TypeCategory::Logical;
}
bool IsIntrinsicLogical(const evaluate::DynamicType &type0, int rank0,
    const evaluate::DynamicType &type1, int rank1) {
  return evaluate::AreConformable(rank0, rank1) &&
      type0.category() == TypeCategory::Logical &&
      type1.category() == TypeCategory::Logical;
}

bool IsIntrinsicConcat(const evaluate::DynamicType &type0, int rank0,
    const evaluate::DynamicType &type1, int rank1) {
  return evaluate::AreConformable(rank0, rank1) &&
      type0.category() == TypeCategory::Character &&
      type1.category() == TypeCategory::Character &&
      type0.kind() == type1.kind();
}

bool IsGenericDefinedOp(const Symbol &symbol) {
  const Symbol &ultimate{symbol.GetUltimate()};
  if (const auto *generic{ultimate.detailsIf<GenericDetails>()}) {
    return generic->kind().IsDefinedOperator();
  } else if (const auto *misc{ultimate.detailsIf<MiscDetails>()}) {
    return misc->kind() == MiscDetails::Kind::TypeBoundDefinedOp;
  } else {
    return false;
  }
}

bool IsCommonBlockContaining(const Symbol &block, const Symbol &object) {
  const auto &objects{block.get<CommonBlockDetails>().objects()};
  auto found{std::find(objects.begin(), objects.end(), object)};
  return found != objects.end();
}

bool IsUseAssociated(const Symbol &symbol, const Scope &scope) {
  const Scope *owner{FindProgramUnitContaining(symbol.GetUltimate().owner())};
  return owner && owner->kind() == Scope::Kind::Module &&
      owner != FindProgramUnitContaining(scope);
}

bool DoesScopeContain(
    const Scope *maybeAncestor, const Scope &maybeDescendent) {
  return maybeAncestor && !maybeDescendent.IsGlobal() &&
      FindScopeContaining(maybeDescendent.parent(),
          [&](const Scope &scope) { return &scope == maybeAncestor; });
}

bool DoesScopeContain(const Scope *maybeAncestor, const Symbol &symbol) {
  return DoesScopeContain(maybeAncestor, symbol.owner());
}

bool IsHostAssociated(const Symbol &symbol, const Scope &scope) {
  const Scope *subprogram{FindProgramUnitContaining(scope)};
  return subprogram &&
      DoesScopeContain(FindProgramUnitContaining(symbol), *subprogram);
}

bool IsInStmtFunction(const Symbol &symbol) {
  if (const Symbol * function{symbol.owner().symbol()}) {
    return IsStmtFunction(*function);
  }
  return false;
}

bool IsStmtFunctionDummy(const Symbol &symbol) {
  return IsDummy(symbol) && IsInStmtFunction(symbol);
}

bool IsStmtFunctionResult(const Symbol &symbol) {
  return IsFunctionResult(symbol) && IsInStmtFunction(symbol);
}

bool IsPointerDummy(const Symbol &symbol) {
  return IsPointer(symbol) && IsDummy(symbol);
}

// proc-name
bool IsProcName(const Symbol &symbol) {
  return symbol.GetUltimate().has<ProcEntityDetails>();
}

bool IsBindCProcedure(const Symbol &symbol) {
  if (const auto *procDetails{symbol.detailsIf<ProcEntityDetails>()}) {
    if (const Symbol * procInterface{procDetails->interface().symbol()}) {
      // procedure component with a BIND(C) interface
      return IsBindCProcedure(*procInterface);
    }
  }
  return symbol.attrs().test(Attr::BIND_C) && IsProcedure(symbol);
}

bool IsBindCProcedure(const Scope &scope) {
  if (const Symbol * symbol{scope.GetSymbol()}) {
    return IsBindCProcedure(*symbol);
  } else {
    return false;
  }
}

static const Symbol *FindPointerComponent(
    const Scope &scope, std::set<const Scope *> &visited) {
  if (!scope.IsDerivedType()) {
    return nullptr;
  }
  if (!visited.insert(&scope).second) {
    return nullptr;
  }
  // If there's a top-level pointer component, return it for clearer error
  // messaging.
  for (const auto &pair : scope) {
    const Symbol &symbol{*pair.second};
    if (IsPointer(symbol)) {
      return &symbol;
    }
  }
  for (const auto &pair : scope) {
    const Symbol &symbol{*pair.second};
    if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
      if (const DeclTypeSpec * type{details->type()}) {
        if (const DerivedTypeSpec * derived{type->AsDerived()}) {
          if (const Scope * nested{derived->scope()}) {
            if (const Symbol *
                pointer{FindPointerComponent(*nested, visited)}) {
              return pointer;
            }
          }
        }
      }
    }
  }
  return nullptr;
}

const Symbol *FindPointerComponent(const Scope &scope) {
  std::set<const Scope *> visited;
  return FindPointerComponent(scope, visited);
}

const Symbol *FindPointerComponent(const DerivedTypeSpec &derived) {
  if (const Scope * scope{derived.scope()}) {
    return FindPointerComponent(*scope);
  } else {
    return nullptr;
  }
}

const Symbol *FindPointerComponent(const DeclTypeSpec &type) {
  if (const DerivedTypeSpec * derived{type.AsDerived()}) {
    return FindPointerComponent(*derived);
  } else {
    return nullptr;
  }
}

const Symbol *FindPointerComponent(const DeclTypeSpec *type) {
  return type ? FindPointerComponent(*type) : nullptr;
}

const Symbol *FindPointerComponent(const Symbol &symbol) {
  return IsPointer(symbol) ? &symbol : FindPointerComponent(symbol.GetType());
}

// C1594 specifies several ways by which an object might be globally visible.
const Symbol *FindExternallyVisibleObject(
    const Symbol &object, const Scope &scope) {
  // TODO: Storage association with any object for which this predicate holds,
  // once EQUIVALENCE is supported.
  if (IsUseAssociated(object, scope) || IsHostAssociated(object, scope) ||
      (IsPureProcedure(scope) && IsPointerDummy(object)) ||
      (IsIntentIn(object) && IsDummy(object))) {
    return &object;
  } else if (const Symbol * block{FindCommonBlockContaining(object)}) {
    return block;
  } else {
    return nullptr;
  }
}

bool ExprHasTypeCategory(
    const SomeExpr &expr, const common::TypeCategory &type) {
  auto dynamicType{expr.GetType()};
  return dynamicType && dynamicType->category() == type;
}

bool ExprTypeKindIsDefault(
    const SomeExpr &expr, const SemanticsContext &context) {
  auto dynamicType{expr.GetType()};
  return dynamicType &&
      dynamicType->category() != common::TypeCategory::Derived &&
      dynamicType->kind() == context.GetDefaultKind(dynamicType->category());
}

// If an analyzed expr or assignment is missing, dump the node and die.
template <typename T>
static void CheckMissingAnalysis(bool absent, const T &x) {
  if (absent) {
    std::string buf;
    llvm::raw_string_ostream ss{buf};
    ss << "node has not been analyzed:\n";
    parser::DumpTree(ss, x);
    common::die(ss.str().c_str());
  }
}

const SomeExpr *GetExprHelper::Get(const parser::Expr &x) {
  CheckMissingAnalysis(!x.typedExpr, x);
  return common::GetPtrFromOptional(x.typedExpr->v);
}
const SomeExpr *GetExprHelper::Get(const parser::Variable &x) {
  CheckMissingAnalysis(!x.typedExpr, x);
  return common::GetPtrFromOptional(x.typedExpr->v);
}
const SomeExpr *GetExprHelper::Get(const parser::DataStmtConstant &x) {
  CheckMissingAnalysis(!x.typedExpr, x);
  return common::GetPtrFromOptional(x.typedExpr->v);
}

const evaluate::Assignment *GetAssignment(const parser::AssignmentStmt &x) {
  CheckMissingAnalysis(!x.typedAssignment, x);
  return common::GetPtrFromOptional(x.typedAssignment->v);
}
const evaluate::Assignment *GetAssignment(
    const parser::PointerAssignmentStmt &x) {
  CheckMissingAnalysis(!x.typedAssignment, x);
  return common::GetPtrFromOptional(x.typedAssignment->v);
}

const Symbol *FindInterface(const Symbol &symbol) {
  return std::visit(
      common::visitors{
          [](const ProcEntityDetails &details) {
            return details.interface().symbol();
          },
          [](const ProcBindingDetails &details) { return &details.symbol(); },
          [](const auto &) -> const Symbol * { return nullptr; },
      },
      symbol.details());
}

const Symbol *FindSubprogram(const Symbol &symbol) {
  return std::visit(
      common::visitors{
          [&](const ProcEntityDetails &details) -> const Symbol * {
            if (const Symbol * interface{details.interface().symbol()}) {
              return FindSubprogram(*interface);
            } else {
              return &symbol;
            }
          },
          [](const ProcBindingDetails &details) {
            return FindSubprogram(details.symbol());
          },
          [&](const SubprogramDetails &) { return &symbol; },
          [](const UseDetails &details) {
            return FindSubprogram(details.symbol());
          },
          [](const HostAssocDetails &details) {
            return FindSubprogram(details.symbol());
          },
          [](const auto &) -> const Symbol * { return nullptr; },
      },
      symbol.details());
}

const Symbol *FindFunctionResult(const Symbol &symbol) {
  if (const Symbol * subp{FindSubprogram(symbol)}) {
    if (const auto &subpDetails{subp->detailsIf<SubprogramDetails>()}) {
      if (subpDetails->isFunction()) {
        return &subpDetails->result();
      }
    }
  }
  return nullptr;
}

const Symbol *FindOverriddenBinding(const Symbol &symbol) {
  if (symbol.has<ProcBindingDetails>()) {
    if (const DeclTypeSpec * parentType{FindParentTypeSpec(symbol.owner())}) {
      if (const DerivedTypeSpec * parentDerived{parentType->AsDerived()}) {
        if (const Scope * parentScope{parentDerived->typeSymbol().scope()}) {
          return parentScope->FindComponent(symbol.name());
        }
      }
    }
  }
  return nullptr;
}

const DeclTypeSpec *FindParentTypeSpec(const DerivedTypeSpec &derived) {
  return FindParentTypeSpec(derived.typeSymbol());
}

const DeclTypeSpec *FindParentTypeSpec(const DeclTypeSpec &decl) {
  if (const DerivedTypeSpec * derived{decl.AsDerived()}) {
    return FindParentTypeSpec(*derived);
  } else {
    return nullptr;
  }
}

const DeclTypeSpec *FindParentTypeSpec(const Scope &scope) {
  if (scope.kind() == Scope::Kind::DerivedType) {
    if (const auto *symbol{scope.symbol()}) {
      return FindParentTypeSpec(*symbol);
    }
  }
  return nullptr;
}

const DeclTypeSpec *FindParentTypeSpec(const Symbol &symbol) {
  if (const Scope * scope{symbol.scope()}) {
    if (const auto *details{symbol.detailsIf<DerivedTypeDetails>()}) {
      if (const Symbol * parent{details->GetParentComponent(*scope)}) {
        return parent->GetType();
      }
    }
  }
  return nullptr;
}

bool IsExtensibleType(const DerivedTypeSpec *derived) {
  return derived && !IsIsoCType(derived) &&
      !derived->typeSymbol().attrs().test(Attr::BIND_C) &&
      !derived->typeSymbol().get<DerivedTypeDetails>().sequence();
}

bool IsBuiltinDerivedType(const DerivedTypeSpec *derived, const char *name) {
  if (!derived) {
    return false;
  } else {
    const auto &symbol{derived->typeSymbol()};
    return symbol.owner().IsModule() &&
        symbol.owner().GetName().value() == "__fortran_builtins" &&
        symbol.name() == "__builtin_"s + name;
  }
}

bool IsIsoCType(const DerivedTypeSpec *derived) {
  return IsBuiltinDerivedType(derived, "c_ptr") ||
      IsBuiltinDerivedType(derived, "c_funptr");
}

bool IsTeamType(const DerivedTypeSpec *derived) {
  return IsBuiltinDerivedType(derived, "team_type");
}

bool IsEventTypeOrLockType(const DerivedTypeSpec *derivedTypeSpec) {
  return IsBuiltinDerivedType(derivedTypeSpec, "event_type") ||
      IsBuiltinDerivedType(derivedTypeSpec, "lock_type");
}

bool IsOrContainsEventOrLockComponent(const Symbol &symbol) {
  if (const Symbol * root{GetAssociationRoot(symbol)}) {
    if (const auto *details{root->detailsIf<ObjectEntityDetails>()}) {
      if (const DeclTypeSpec * type{details->type()}) {
        if (const DerivedTypeSpec * derived{type->AsDerived()}) {
          return IsEventTypeOrLockType(derived) ||
              FindEventOrLockPotentialComponent(*derived);
        }
      }
    }
  }
  return false;
}

// Check this symbol suitable as a type-bound procedure - C769
bool CanBeTypeBoundProc(const Symbol *symbol) {
  if (!symbol || IsDummy(*symbol) || IsProcedurePointer(*symbol)) {
    return false;
  } else if (symbol->has<SubprogramNameDetails>()) {
    return symbol->owner().kind() == Scope::Kind::Module;
  } else if (auto *details{symbol->detailsIf<SubprogramDetails>()}) {
    return symbol->owner().kind() == Scope::Kind::Module ||
        details->isInterface();
  } else if (const auto *proc{symbol->detailsIf<ProcEntityDetails>()}) {
    return !symbol->attrs().test(Attr::INTRINSIC) &&
        proc->HasExplicitInterface();
  } else {
    return false;
  }
}

bool IsInitialized(const Symbol &symbol, bool ignoreDATAstatements) {
  if (!ignoreDATAstatements && symbol.test(Symbol::Flag::InDataStmt)) {
    return true;
  } else if (IsNamedConstant(symbol)) {
    return false;
  } else if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (object->init()) {
      return true;
    } else if (object->isDummy() || IsFunctionResult(symbol)) {
      return false;
    } else if (IsAllocatable(symbol)) {
      return true;
    } else if (!IsPointer(symbol) && object->type()) {
      if (const auto *derived{object->type()->AsDerived()}) {
        if (derived->HasDefaultInitialization()) {
          return true;
        }
      }
    }
  } else if (const auto *proc{symbol.detailsIf<ProcEntityDetails>()}) {
    return proc->init().has_value();
  }
  return false;
}

bool HasIntrinsicTypeName(const Symbol &symbol) {
  std::string name{symbol.name().ToString()};
  if (name == "doubleprecision") {
    return true;
  } else if (name == "derived") {
    return false;
  } else {
    for (int i{0}; i != common::TypeCategory_enumSize; ++i) {
      if (name == parser::ToLowerCaseLetters(EnumToString(TypeCategory{i}))) {
        return true;
      }
    }
    return false;
  }
}

bool IsSeparateModuleProcedureInterface(const Symbol *symbol) {
  if (symbol && symbol->attrs().test(Attr::MODULE)) {
    if (auto *details{symbol->detailsIf<SubprogramDetails>()}) {
      return details->isInterface();
    }
  }
  return false;
}

// 3.11 automatic data object
bool IsAutomatic(const Symbol &symbol) {
  if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (!object->isDummy() && !IsAllocatable(symbol) && !IsPointer(symbol)) {
      if (const DeclTypeSpec * type{symbol.GetType()}) {
        // If a type parameter value is not a constant expression, the
        // object is automatic.
        if (type->category() == DeclTypeSpec::Character) {
          if (const auto &length{
                  type->characterTypeSpec().length().GetExplicit()}) {
            if (!evaluate::IsConstantExpr(*length)) {
              return true;
            }
          }
        } else if (const DerivedTypeSpec * derived{type->AsDerived()}) {
          for (const auto &pair : derived->parameters()) {
            if (const auto &value{pair.second.GetExplicit()}) {
              if (!evaluate::IsConstantExpr(*value)) {
                return true;
              }
            }
          }
        }
      }
      // If an array bound is not a constant expression, the object is
      // automatic.
      for (const ShapeSpec &dim : object->shape()) {
        if (const auto &lb{dim.lbound().GetExplicit()}) {
          if (!evaluate::IsConstantExpr(*lb)) {
            return true;
          }
        }
        if (const auto &ub{dim.ubound().GetExplicit()}) {
          if (!evaluate::IsConstantExpr(*ub)) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

bool IsFinalizable(const Symbol &symbol) {
  if (const DeclTypeSpec * type{symbol.GetType()}) {
    if (const DerivedTypeSpec * derived{type->AsDerived()}) {
      return IsFinalizable(*derived);
    }
  }
  return false;
}

bool IsFinalizable(const DerivedTypeSpec &derived) {
  ScopeComponentIterator components{derived};
  return std::find_if(components.begin(), components.end(),
             [](const Symbol &x) { return x.has<FinalProcDetails>(); }) !=
      components.end();
}

// TODO The following function returns true for all types with FINAL procedures
// This is because we don't yet fill in the data for FinalProcDetails
bool HasImpureFinal(const DerivedTypeSpec &derived) {
  ScopeComponentIterator components{derived};
  return std::find_if(
             components.begin(), components.end(), [](const Symbol &x) {
               return x.has<FinalProcDetails>() && !x.attrs().test(Attr::PURE);
             }) != components.end();
}

bool IsCoarray(const Symbol &symbol) { return symbol.Corank() > 0; }

bool IsAutomaticObject(const Symbol &symbol) {
  if (IsDummy(symbol) || IsPointer(symbol) || IsAllocatable(symbol)) {
    return false;
  }
  if (const DeclTypeSpec * type{symbol.GetType()}) {
    if (type->category() == DeclTypeSpec::Character) {
      ParamValue length{type->characterTypeSpec().length()};
      if (length.isExplicit()) {
        if (MaybeIntExpr lengthExpr{length.GetExplicit()}) {
          if (!ToInt64(lengthExpr)) {
            return true;
          }
        }
      }
    }
  }
  if (symbol.IsObjectArray()) {
    for (const ShapeSpec &spec : symbol.get<ObjectEntityDetails>().shape()) {
      auto &lbound{spec.lbound().GetExplicit()};
      auto &ubound{spec.ubound().GetExplicit()};
      if ((lbound && !evaluate::ToInt64(*lbound)) ||
          (ubound && !evaluate::ToInt64(*ubound))) {
        return true;
      }
    }
  }
  return false;
}

bool IsAssumedLengthCharacter(const Symbol &symbol) {
  if (const DeclTypeSpec * type{symbol.GetType()}) {
    return type->category() == DeclTypeSpec::Character &&
        type->characterTypeSpec().length().isAssumed();
  } else {
    return false;
  }
}

bool IsInBlankCommon(const Symbol &symbol) {
  const Symbol *block{FindCommonBlockContaining(symbol)};
  return block && block->name().empty();
}

// C722 and C723:  For a function to be assumed length, it must be external and
// of CHARACTER type
bool IsExternal(const Symbol &symbol) {
  return (symbol.has<SubprogramDetails>() && symbol.owner().IsGlobal()) ||
      symbol.attrs().test(Attr::EXTERNAL);
}

const Symbol *IsExternalInPureContext(
    const Symbol &symbol, const Scope &scope) {
  if (const auto *pureProc{FindPureProcedureContaining(scope)}) {
    if (const Symbol * root{GetAssociationRoot(symbol)}) {
      if (const Symbol *
          visible{FindExternallyVisibleObject(*root, *pureProc)}) {
        return visible;
      }
    }
  }
  return nullptr;
}

PotentialComponentIterator::const_iterator FindPolymorphicPotentialComponent(
    const DerivedTypeSpec &derived) {
  PotentialComponentIterator potentials{derived};
  return std::find_if(
      potentials.begin(), potentials.end(), [](const Symbol &component) {
        if (const auto *details{component.detailsIf<ObjectEntityDetails>()}) {
          const DeclTypeSpec *type{details->type()};
          return type && type->IsPolymorphic();
        }
        return false;
      });
}

bool IsOrContainsPolymorphicComponent(const Symbol &symbol) {
  if (const Symbol * root{GetAssociationRoot(symbol)}) {
    if (const auto *details{root->detailsIf<ObjectEntityDetails>()}) {
      if (const DeclTypeSpec * type{details->type()}) {
        if (type->IsPolymorphic()) {
          return true;
        }
        if (const DerivedTypeSpec * derived{type->AsDerived()}) {
          return (bool)FindPolymorphicPotentialComponent(*derived);
        }
      }
    }
  }
  return false;
}

bool InProtectedContext(const Symbol &symbol, const Scope &currentScope) {
  return IsProtected(symbol) && !IsHostAssociated(symbol, currentScope);
}

// C1101 and C1158
// TODO Need to check for a coindexed object (why? C1103?)
std::optional<parser::MessageFixedText> WhyNotModifiable(
    const Symbol &symbol, const Scope &scope) {
  const Symbol *root{GetAssociationRoot(symbol)};
  if (!root) {
    return "'%s' is construct associated with an expression"_en_US;
  } else if (InProtectedContext(*root, scope)) {
    return "'%s' is protected in this scope"_en_US;
  } else if (IsExternalInPureContext(*root, scope)) {
    return "'%s' is externally visible and referenced in a pure"
           " procedure"_en_US;
  } else if (IsOrContainsEventOrLockComponent(*root)) {
    return "'%s' is an entity with either an EVENT_TYPE or LOCK_TYPE"_en_US;
  } else if (IsIntentIn(*root)) {
    return "'%s' is an INTENT(IN) dummy argument"_en_US;
  } else if (!IsVariableName(*root)) {
    return "'%s' is not a variable"_en_US;
  } else {
    return std::nullopt;
  }
}

std::optional<parser::Message> WhyNotModifiable(parser::CharBlock at,
    const SomeExpr &expr, const Scope &scope, bool vectorSubscriptIsOk) {
  if (!evaluate::IsVariable(expr)) {
    return parser::Message{at, "Expression is not a variable"_en_US};
  } else if (auto dataRef{evaluate::ExtractDataRef(expr, true)}) {
    if (!vectorSubscriptIsOk && evaluate::HasVectorSubscript(expr)) {
      return parser::Message{at, "Variable has a vector subscript"_en_US};
    }
    const Symbol &symbol{dataRef->GetFirstSymbol()};
    if (auto maybeWhy{WhyNotModifiable(symbol, scope)}) {
      return parser::Message{symbol.name(),
          parser::MessageFormattedText{std::move(*maybeWhy), symbol.name()}};
    }
  } else {
    // reference to function returning POINTER
  }
  return std::nullopt;
}

class ImageControlStmtHelper {
  using ImageControlStmts = std::variant<parser::ChangeTeamConstruct,
      parser::CriticalConstruct, parser::EventPostStmt, parser::EventWaitStmt,
      parser::FormTeamStmt, parser::LockStmt, parser::StopStmt,
      parser::SyncAllStmt, parser::SyncImagesStmt, parser::SyncMemoryStmt,
      parser::SyncTeamStmt, parser::UnlockStmt>;

public:
  template <typename T> bool operator()(const T &) {
    return common::HasMember<T, ImageControlStmts>;
  }
  template <typename T> bool operator()(const common::Indirection<T> &x) {
    return (*this)(x.value());
  }
  bool operator()(const parser::AllocateStmt &stmt) {
    const auto &allocationList{std::get<std::list<parser::Allocation>>(stmt.t)};
    for (const auto &allocation : allocationList) {
      const auto &allocateObject{
          std::get<parser::AllocateObject>(allocation.t)};
      if (IsCoarrayObject(allocateObject)) {
        return true;
      }
    }
    return false;
  }
  bool operator()(const parser::DeallocateStmt &stmt) {
    const auto &allocateObjectList{
        std::get<std::list<parser::AllocateObject>>(stmt.t)};
    for (const auto &allocateObject : allocateObjectList) {
      if (IsCoarrayObject(allocateObject)) {
        return true;
      }
    }
    return false;
  }
  bool operator()(const parser::CallStmt &stmt) {
    const auto &procedureDesignator{
        std::get<parser::ProcedureDesignator>(stmt.v.t)};
    if (auto *name{std::get_if<parser::Name>(&procedureDesignator.u)}) {
      // TODO: also ensure that the procedure is, in fact, an intrinsic
      if (name->source == "move_alloc") {
        const auto &args{std::get<std::list<parser::ActualArgSpec>>(stmt.v.t)};
        if (!args.empty()) {
          const parser::ActualArg &actualArg{
              std::get<parser::ActualArg>(args.front().t)};
          if (const auto *argExpr{
                  std::get_if<common::Indirection<parser::Expr>>(
                      &actualArg.u)}) {
            return HasCoarray(argExpr->value());
          }
        }
      }
    }
    return false;
  }
  bool operator()(const parser::Statement<parser::ActionStmt> &stmt) {
    return std::visit(*this, stmt.statement.u);
  }

private:
  bool IsCoarrayObject(const parser::AllocateObject &allocateObject) {
    const parser::Name &name{GetLastName(allocateObject)};
    return name.symbol && IsCoarray(*name.symbol);
  }
};

bool IsImageControlStmt(const parser::ExecutableConstruct &construct) {
  return std::visit(ImageControlStmtHelper{}, construct.u);
}

std::optional<parser::MessageFixedText> GetImageControlStmtCoarrayMsg(
    const parser::ExecutableConstruct &construct) {
  if (const auto *actionStmt{
          std::get_if<parser::Statement<parser::ActionStmt>>(&construct.u)}) {
    return std::visit(
        common::visitors{
            [](const common::Indirection<parser::AllocateStmt> &)
                -> std::optional<parser::MessageFixedText> {
              return "ALLOCATE of a coarray is an image control"
                     " statement"_en_US;
            },
            [](const common::Indirection<parser::DeallocateStmt> &)
                -> std::optional<parser::MessageFixedText> {
              return "DEALLOCATE of a coarray is an image control"
                     " statement"_en_US;
            },
            [](const common::Indirection<parser::CallStmt> &)
                -> std::optional<parser::MessageFixedText> {
              return "MOVE_ALLOC of a coarray is an image control"
                     " statement "_en_US;
            },
            [](const auto &) -> std::optional<parser::MessageFixedText> {
              return std::nullopt;
            },
        },
        actionStmt->statement.u);
  }
  return std::nullopt;
}

parser::CharBlock GetImageControlStmtLocation(
    const parser::ExecutableConstruct &executableConstruct) {
  return std::visit(
      common::visitors{
          [](const common::Indirection<parser::ChangeTeamConstruct>
                  &construct) {
            return std::get<parser::Statement<parser::ChangeTeamStmt>>(
                construct.value().t)
                .source;
          },
          [](const common::Indirection<parser::CriticalConstruct> &construct) {
            return std::get<parser::Statement<parser::CriticalStmt>>(
                construct.value().t)
                .source;
          },
          [](const parser::Statement<parser::ActionStmt> &actionStmt) {
            return actionStmt.source;
          },
          [](const auto &) { return parser::CharBlock{}; },
      },
      executableConstruct.u);
}

bool HasCoarray(const parser::Expr &expression) {
  if (const auto *expr{GetExpr(expression)}) {
    for (const Symbol &symbol : evaluate::CollectSymbols(*expr)) {
      if (const Symbol * root{GetAssociationRoot(symbol)}) {
        if (IsCoarray(*root)) {
          return true;
        }
      }
    }
  }
  return false;
}

bool IsPolymorphic(const Symbol &symbol) {
  if (const DeclTypeSpec * type{symbol.GetType()}) {
    return type->IsPolymorphic();
  }
  return false;
}

bool IsPolymorphicAllocatable(const Symbol &symbol) {
  return IsAllocatable(symbol) && IsPolymorphic(symbol);
}

std::optional<parser::MessageFormattedText> CheckAccessibleComponent(
    const Scope &scope, const Symbol &symbol) {
  CHECK(symbol.owner().IsDerivedType()); // symbol must be a component
  if (symbol.attrs().test(Attr::PRIVATE)) {
    if (const Scope * moduleScope{FindModuleContaining(symbol.owner())}) {
      if (!moduleScope->Contains(scope)) {
        return parser::MessageFormattedText{
            "PRIVATE component '%s' is only accessible within module '%s'"_err_en_US,
            symbol.name(), moduleScope->GetName().value()};
      }
    }
  }
  return std::nullopt;
}

std::list<SourceName> OrderParameterNames(const Symbol &typeSymbol) {
  std::list<SourceName> result;
  if (const DerivedTypeSpec * spec{typeSymbol.GetParentTypeSpec()}) {
    result = OrderParameterNames(spec->typeSymbol());
  }
  const auto &paramNames{typeSymbol.get<DerivedTypeDetails>().paramNames()};
  result.insert(result.end(), paramNames.begin(), paramNames.end());
  return result;
}

SymbolVector OrderParameterDeclarations(const Symbol &typeSymbol) {
  SymbolVector result;
  if (const DerivedTypeSpec * spec{typeSymbol.GetParentTypeSpec()}) {
    result = OrderParameterDeclarations(spec->typeSymbol());
  }
  const auto &paramDecls{typeSymbol.get<DerivedTypeDetails>().paramDecls()};
  result.insert(result.end(), paramDecls.begin(), paramDecls.end());
  return result;
}

const DeclTypeSpec &FindOrInstantiateDerivedType(Scope &scope,
    DerivedTypeSpec &&spec, SemanticsContext &semanticsContext,
    DeclTypeSpec::Category category) {
  spec.CookParameters(semanticsContext.foldingContext());
  spec.EvaluateParameters(semanticsContext.foldingContext());
  if (const DeclTypeSpec *
      type{scope.FindInstantiatedDerivedType(spec, category)}) {
    return *type;
  }
  // Create a new instantiation of this parameterized derived type
  // for this particular distinct set of actual parameter values.
  DeclTypeSpec &type{scope.MakeDerivedType(category, std::move(spec))};
  type.derivedTypeSpec().Instantiate(scope, semanticsContext);
  return type;
}

const Symbol *FindSeparateModuleSubprogramInterface(const Symbol *proc) {
  if (proc) {
    if (const Symbol * submodule{proc->owner().symbol()}) {
      if (const auto *details{submodule->detailsIf<ModuleDetails>()}) {
        if (const Scope * ancestor{details->ancestor()}) {
          const Symbol *iface{ancestor->FindSymbol(proc->name())};
          if (IsSeparateModuleProcedureInterface(iface)) {
            return iface;
          }
        }
      }
    }
  }
  return nullptr;
}

// ComponentIterator implementation

template <ComponentKind componentKind>
typename ComponentIterator<componentKind>::const_iterator
ComponentIterator<componentKind>::const_iterator::Create(
    const DerivedTypeSpec &derived) {
  const_iterator it{};
  it.componentPath_.emplace_back(derived);
  it.Increment(); // cue up first relevant component, if any
  return it;
}

template <ComponentKind componentKind>
const DerivedTypeSpec *
ComponentIterator<componentKind>::const_iterator::PlanComponentTraversal(
    const Symbol &component) const {
  if (const auto *details{component.detailsIf<ObjectEntityDetails>()}) {
    if (const DeclTypeSpec * type{details->type()}) {
      if (const auto *derived{type->AsDerived()}) {
        bool traverse{false};
        if constexpr (componentKind == ComponentKind::Ordered) {
          // Order Component (only visit parents)
          traverse = component.test(Symbol::Flag::ParentComp);
        } else if constexpr (componentKind == ComponentKind::Direct) {
          traverse = !IsAllocatableOrPointer(component);
        } else if constexpr (componentKind == ComponentKind::Ultimate) {
          traverse = !IsAllocatableOrPointer(component);
        } else if constexpr (componentKind == ComponentKind::Potential) {
          traverse = !IsPointer(component);
        } else if constexpr (componentKind == ComponentKind::Scope) {
          traverse = !IsAllocatableOrPointer(component);
        }
        if (traverse) {
          const Symbol &newTypeSymbol{derived->typeSymbol()};
          // Avoid infinite loop if the type is already part of the types
          // being visited. It is possible to have "loops in type" because
          // C744 does not forbid to use not yet declared type for
          // ALLOCATABLE or POINTER components.
          for (const auto &node : componentPath_) {
            if (&newTypeSymbol == &node.GetTypeSymbol()) {
              return nullptr;
            }
          }
          return derived;
        }
      }
    } // intrinsic & unlimited polymorphic not traversable
  }
  return nullptr;
}

template <ComponentKind componentKind>
static bool StopAtComponentPre(const Symbol &component) {
  if constexpr (componentKind == ComponentKind::Ordered) {
    // Parent components need to be iterated upon after their
    // sub-components in structure constructor analysis.
    return !component.test(Symbol::Flag::ParentComp);
  } else if constexpr (componentKind == ComponentKind::Direct) {
    return true;
  } else if constexpr (componentKind == ComponentKind::Ultimate) {
    return component.has<ProcEntityDetails>() ||
        IsAllocatableOrPointer(component) ||
        (component.get<ObjectEntityDetails>().type() &&
            component.get<ObjectEntityDetails>().type()->AsIntrinsic());
  } else if constexpr (componentKind == ComponentKind::Potential) {
    return !IsPointer(component);
  }
}

template <ComponentKind componentKind>
static bool StopAtComponentPost(const Symbol &component) {
  return componentKind == ComponentKind::Ordered &&
      component.test(Symbol::Flag::ParentComp);
}

template <ComponentKind componentKind>
void ComponentIterator<componentKind>::const_iterator::Increment() {
  while (!componentPath_.empty()) {
    ComponentPathNode &deepest{componentPath_.back()};
    if (deepest.component()) {
      if (!deepest.descended()) {
        deepest.set_descended(true);
        if (const DerivedTypeSpec *
            derived{PlanComponentTraversal(*deepest.component())}) {
          componentPath_.emplace_back(*derived);
          continue;
        }
      } else if (!deepest.visited()) {
        deepest.set_visited(true);
        return; // this is the next component to visit, after descending
      }
    }
    auto &nameIterator{deepest.nameIterator()};
    if (nameIterator == deepest.nameEnd()) {
      componentPath_.pop_back();
    } else if constexpr (componentKind == ComponentKind::Scope) {
      deepest.set_component(*nameIterator++->second);
      deepest.set_descended(false);
      deepest.set_visited(true);
      return; // this is the next component to visit, before descending
    } else {
      const Scope &scope{deepest.GetScope()};
      auto scopeIter{scope.find(*nameIterator++)};
      if (scopeIter != scope.cend()) {
        const Symbol &component{*scopeIter->second};
        deepest.set_component(component);
        deepest.set_descended(false);
        if (StopAtComponentPre<componentKind>(component)) {
          deepest.set_visited(true);
          return; // this is the next component to visit, before descending
        } else {
          deepest.set_visited(!StopAtComponentPost<componentKind>(component));
        }
      }
    }
  }
}

template <ComponentKind componentKind>
std::string
ComponentIterator<componentKind>::const_iterator::BuildResultDesignatorName()
    const {
  std::string designator{""};
  for (const auto &node : componentPath_) {
    designator += "%" + DEREF(node.component()).name().ToString();
  }
  return designator;
}

template class ComponentIterator<ComponentKind::Ordered>;
template class ComponentIterator<ComponentKind::Direct>;
template class ComponentIterator<ComponentKind::Ultimate>;
template class ComponentIterator<ComponentKind::Potential>;
template class ComponentIterator<ComponentKind::Scope>;

UltimateComponentIterator::const_iterator FindCoarrayUltimateComponent(
    const DerivedTypeSpec &derived) {
  UltimateComponentIterator ultimates{derived};
  return std::find_if(ultimates.begin(), ultimates.end(), IsCoarray);
}

UltimateComponentIterator::const_iterator FindPointerUltimateComponent(
    const DerivedTypeSpec &derived) {
  UltimateComponentIterator ultimates{derived};
  return std::find_if(ultimates.begin(), ultimates.end(), IsPointer);
}

PotentialComponentIterator::const_iterator FindEventOrLockPotentialComponent(
    const DerivedTypeSpec &derived) {
  PotentialComponentIterator potentials{derived};
  return std::find_if(
      potentials.begin(), potentials.end(), [](const Symbol &component) {
        if (const auto *details{component.detailsIf<ObjectEntityDetails>()}) {
          const DeclTypeSpec *type{details->type()};
          return type && IsEventTypeOrLockType(type->AsDerived());
        }
        return false;
      });
}

UltimateComponentIterator::const_iterator FindAllocatableUltimateComponent(
    const DerivedTypeSpec &derived) {
  UltimateComponentIterator ultimates{derived};
  return std::find_if(ultimates.begin(), ultimates.end(), IsAllocatable);
}

UltimateComponentIterator::const_iterator
FindPolymorphicAllocatableUltimateComponent(const DerivedTypeSpec &derived) {
  UltimateComponentIterator ultimates{derived};
  return std::find_if(
      ultimates.begin(), ultimates.end(), IsPolymorphicAllocatable);
}

UltimateComponentIterator::const_iterator
FindPolymorphicAllocatableNonCoarrayUltimateComponent(
    const DerivedTypeSpec &derived) {
  UltimateComponentIterator ultimates{derived};
  return std::find_if(ultimates.begin(), ultimates.end(), [](const Symbol &x) {
    return IsPolymorphicAllocatable(x) && !IsCoarray(x);
  });
}

const Symbol *FindUltimateComponent(const DerivedTypeSpec &derived,
    const std::function<bool(const Symbol &)> &predicate) {
  UltimateComponentIterator ultimates{derived};
  if (auto it{std::find_if(ultimates.begin(), ultimates.end(),
          [&predicate](const Symbol &component) -> bool {
            return predicate(component);
          })}) {
    return &*it;
  }
  return nullptr;
}

const Symbol *FindUltimateComponent(const Symbol &symbol,
    const std::function<bool(const Symbol &)> &predicate) {
  if (predicate(symbol)) {
    return &symbol;
  } else if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (const auto *type{object->type()}) {
      if (const auto *derived{type->AsDerived()}) {
        return FindUltimateComponent(*derived, predicate);
      }
    }
  }
  return nullptr;
}

const Symbol *FindImmediateComponent(const DerivedTypeSpec &type,
    const std::function<bool(const Symbol &)> &predicate) {
  if (const Scope * scope{type.scope()}) {
    const Symbol *parent{nullptr};
    for (const auto &pair : *scope) {
      const Symbol *symbol{&*pair.second};
      if (predicate(*symbol)) {
        return symbol;
      }
      if (symbol->test(Symbol::Flag::ParentComp)) {
        parent = symbol;
      }
    }
    if (parent) {
      if (const auto *object{parent->detailsIf<ObjectEntityDetails>()}) {
        if (const auto *type{object->type()}) {
          if (const auto *derived{type->AsDerived()}) {
            return FindImmediateComponent(*derived, predicate);
          }
        }
      }
    }
  }
  return nullptr;
}

bool IsFunctionResult(const Symbol &symbol) {
  return (symbol.has<ObjectEntityDetails>() &&
             symbol.get<ObjectEntityDetails>().isFuncResult()) ||
      (symbol.has<ProcEntityDetails>() &&
          symbol.get<ProcEntityDetails>().isFuncResult());
}

bool IsFunctionResultWithSameNameAsFunction(const Symbol &symbol) {
  if (IsFunctionResult(symbol)) {
    if (const Symbol * function{symbol.owner().symbol()}) {
      return symbol.name() == function->name();
    }
  }
  return false;
}

void LabelEnforce::Post(const parser::GotoStmt &gotoStmt) {
  checkLabelUse(gotoStmt.v);
}
void LabelEnforce::Post(const parser::ComputedGotoStmt &computedGotoStmt) {
  for (auto &i : std::get<std::list<parser::Label>>(computedGotoStmt.t)) {
    checkLabelUse(i);
  }
}

void LabelEnforce::Post(const parser::ArithmeticIfStmt &arithmeticIfStmt) {
  checkLabelUse(std::get<1>(arithmeticIfStmt.t));
  checkLabelUse(std::get<2>(arithmeticIfStmt.t));
  checkLabelUse(std::get<3>(arithmeticIfStmt.t));
}

void LabelEnforce::Post(const parser::AssignStmt &assignStmt) {
  checkLabelUse(std::get<parser::Label>(assignStmt.t));
}

void LabelEnforce::Post(const parser::AssignedGotoStmt &assignedGotoStmt) {
  for (auto &i : std::get<std::list<parser::Label>>(assignedGotoStmt.t)) {
    checkLabelUse(i);
  }
}

void LabelEnforce::Post(const parser::AltReturnSpec &altReturnSpec) {
  checkLabelUse(altReturnSpec.v);
}

void LabelEnforce::Post(const parser::ErrLabel &errLabel) {
  checkLabelUse(errLabel.v);
}
void LabelEnforce::Post(const parser::EndLabel &endLabel) {
  checkLabelUse(endLabel.v);
}
void LabelEnforce::Post(const parser::EorLabel &eorLabel) {
  checkLabelUse(eorLabel.v);
}

void LabelEnforce::checkLabelUse(const parser::Label &labelUsed) {
  if (labels_.find(labelUsed) == labels_.end()) {
    SayWithConstruct(context_, currentStatementSourcePosition_,
        parser::MessageFormattedText{
            "Control flow escapes from %s"_err_en_US, construct_},
        constructSourcePosition_);
  }
}

parser::MessageFormattedText LabelEnforce::GetEnclosingConstructMsg() {
  return {"Enclosing %s statement"_en_US, construct_};
}

void LabelEnforce::SayWithConstruct(SemanticsContext &context,
    parser::CharBlock stmtLocation, parser::MessageFormattedText &&message,
    parser::CharBlock constructLocation) {
  context.Say(stmtLocation, message)
      .Attach(constructLocation, GetEnclosingConstructMsg());
}

} // namespace Fortran::semantics
