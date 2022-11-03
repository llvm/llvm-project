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
    if (scope->IsTopLevel()) {
      return nullptr;
    }
  }
}

const Scope &GetTopLevelUnitContaining(const Scope &start) {
  CHECK(!start.IsTopLevel());
  return DEREF(FindScopeContaining(
      start, [](const Scope &scope) { return scope.parent().IsTopLevel(); }));
}

const Scope &GetTopLevelUnitContaining(const Symbol &symbol) {
  return GetTopLevelUnitContaining(symbol.owner());
}

const Scope *FindModuleContaining(const Scope &start) {
  return FindScopeContaining(
      start, [](const Scope &scope) { return scope.IsModule(); });
}

const Scope *FindModuleFileContaining(const Scope &start) {
  return FindScopeContaining(
      start, [](const Scope &scope) { return scope.IsModuleFile(); });
}

const Scope &GetProgramUnitContaining(const Scope &start) {
  CHECK(!start.IsTopLevel());
  return DEREF(FindScopeContaining(start, [](const Scope &scope) {
    switch (scope.kind()) {
    case Scope::Kind::Module:
    case Scope::Kind::MainProgram:
    case Scope::Kind::Subprogram:
    case Scope::Kind::BlockData:
      return true;
    default:
      return false;
    }
  }));
}

const Scope &GetProgramUnitContaining(const Symbol &symbol) {
  return GetProgramUnitContaining(symbol.owner());
}

const Scope &GetProgramUnitOrBlockConstructContaining(const Scope &start) {
  CHECK(!start.IsTopLevel());
  return DEREF(FindScopeContaining(start, [](const Scope &scope) {
    switch (scope.kind()) {
    case Scope::Kind::Module:
    case Scope::Kind::MainProgram:
    case Scope::Kind::Subprogram:
    case Scope::Kind::BlockData:
    case Scope::Kind::BlockConstruct:
      return true;
    default:
      return false;
    }
  }));
}

const Scope &GetProgramUnitOrBlockConstructContaining(const Symbol &symbol) {
  return GetProgramUnitOrBlockConstructContaining(symbol.owner());
}

const Scope *FindPureProcedureContaining(const Scope &start) {
  // N.B. We only need to examine the innermost containing program unit
  // because an internal subprogram of a pure subprogram must also
  // be pure (C1592).
  if (start.IsTopLevel()) {
    return nullptr;
  } else {
    const Scope &scope{GetProgramUnitContaining(start)};
    return IsPureProcedure(scope) ? &scope : nullptr;
  }
}

// 7.5.2.4 "same derived type" test -- rely on IsTkCompatibleWith() and its
// infrastructure to detect and handle comparisons on distinct (but "same")
// sequence/bind(C) derived types
static bool MightBeSameDerivedType(
    const std::optional<evaluate::DynamicType> &lhsType,
    const std::optional<evaluate::DynamicType> &rhsType) {
  return lhsType && rhsType && lhsType->IsTkCompatibleWith(*rhsType);
}

Tristate IsDefinedAssignment(
    const std::optional<evaluate::DynamicType> &lhsType, int lhsRank,
    const std::optional<evaluate::DynamicType> &rhsType, int rhsRank) {
  if (!lhsType || !rhsType) {
    return Tristate::No; // error or rhs is untyped
  }
  if (lhsType->IsUnlimitedPolymorphic()) {
    return Tristate::No;
  }
  if (rhsType->IsUnlimitedPolymorphic()) {
    return Tristate::Maybe;
  }
  TypeCategory lhsCat{lhsType->category()};
  TypeCategory rhsCat{rhsType->category()};
  if (rhsRank > 0 && lhsRank != rhsRank) {
    return Tristate::Yes;
  } else if (lhsCat != TypeCategory::Derived) {
    return ToTristate(lhsCat != rhsCat &&
        (!IsNumericTypeCategory(lhsCat) || !IsNumericTypeCategory(rhsCat)));
  } else if (MightBeSameDerivedType(lhsType, rhsType)) {
    return Tristate::Maybe; // TYPE(t) = TYPE(t) can be defined or intrinsic
  } else {
    return Tristate::Yes;
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

bool IsDefinedOperator(SourceName name) {
  const char *begin{name.begin()};
  const char *end{name.end()};
  return begin != end && begin[0] == '.' && end[-1] == '.';
}

std::string MakeOpName(SourceName name) {
  std::string result{name.ToString()};
  return IsDefinedOperator(name)         ? "OPERATOR(" + result + ")"
      : result.find("operator(", 0) == 0 ? parser::ToUpperCaseLetters(result)
                                         : result;
}

bool IsCommonBlockContaining(const Symbol &block, const Symbol &object) {
  const auto &objects{block.get<CommonBlockDetails>().objects()};
  return llvm::is_contained(objects, object);
}

bool IsUseAssociated(const Symbol &symbol, const Scope &scope) {
  const Scope &owner{GetTopLevelUnitContaining(symbol.GetUltimate().owner())};
  return owner.kind() == Scope::Kind::Module &&
      owner != GetTopLevelUnitContaining(scope);
}

bool DoesScopeContain(
    const Scope *maybeAncestor, const Scope &maybeDescendent) {
  return maybeAncestor && !maybeDescendent.IsTopLevel() &&
      FindScopeContaining(maybeDescendent.parent(),
          [&](const Scope &scope) { return &scope == maybeAncestor; });
}

bool DoesScopeContain(const Scope *maybeAncestor, const Symbol &symbol) {
  return DoesScopeContain(maybeAncestor, symbol.owner());
}

static const Symbol &FollowHostAssoc(const Symbol &symbol) {
  for (const Symbol *s{&symbol};;) {
    const auto *details{s->detailsIf<HostAssocDetails>()};
    if (!details) {
      return *s;
    }
    s = &details->symbol();
  }
}

bool IsHostAssociated(const Symbol &symbol, const Scope &scope) {
  return DoesScopeContain(
      &GetProgramUnitOrBlockConstructContaining(FollowHostAssoc(symbol)),
      GetProgramUnitOrBlockConstructContaining(scope));
}

bool IsHostAssociatedIntoSubprogram(const Symbol &symbol, const Scope &scope) {
  return DoesScopeContain(
      &GetProgramUnitOrBlockConstructContaining(FollowHostAssoc(symbol)),
      GetProgramUnitContaining(scope));
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
    const Symbol &object, const Scope &scope, bool isPointerDefinition) {
  // TODO: Storage association with any object for which this predicate holds,
  // once EQUIVALENCE is supported.
  const Symbol &ultimate{GetAssociationRoot(object)};
  if (IsDummy(ultimate)) {
    if (IsIntentIn(ultimate)) {
      return &ultimate;
    }
    if (!isPointerDefinition && IsPointer(ultimate) &&
        IsPureProcedure(ultimate.owner()) && IsFunction(ultimate.owner())) {
      return &ultimate;
    }
  } else if (ultimate.owner().IsDerivedType()) {
    return nullptr;
  } else if (&GetProgramUnitContaining(ultimate) !=
      &GetProgramUnitContaining(scope)) {
    return &object;
  } else if (const Symbol * block{FindCommonBlockContaining(ultimate)}) {
    return block;
  }
  return nullptr;
}

const Symbol &BypassGeneric(const Symbol &symbol) {
  const Symbol &ultimate{symbol.GetUltimate()};
  if (const auto *generic{ultimate.detailsIf<GenericDetails>()}) {
    if (const Symbol * specific{generic->specific()}) {
      return *specific;
    }
  }
  return symbol;
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
static void CheckMissingAnalysis(
    bool crash, SemanticsContext *context, const T &x) {
  if (crash && !(context && context->AnyFatalError())) {
    std::string buf;
    llvm::raw_string_ostream ss{buf};
    ss << "node has not been analyzed:\n";
    parser::DumpTree(ss, x);
    common::die(ss.str().c_str());
  }
}

const SomeExpr *GetExprHelper::Get(const parser::Expr &x) {
  CheckMissingAnalysis(crashIfNoExpr_ && !x.typedExpr, context_, x);
  return x.typedExpr ? common::GetPtrFromOptional(x.typedExpr->v) : nullptr;
}
const SomeExpr *GetExprHelper::Get(const parser::Variable &x) {
  CheckMissingAnalysis(crashIfNoExpr_ && !x.typedExpr, context_, x);
  return x.typedExpr ? common::GetPtrFromOptional(x.typedExpr->v) : nullptr;
}
const SomeExpr *GetExprHelper::Get(const parser::DataStmtConstant &x) {
  CheckMissingAnalysis(crashIfNoExpr_ && !x.typedExpr, context_, x);
  return x.typedExpr ? common::GetPtrFromOptional(x.typedExpr->v) : nullptr;
}
const SomeExpr *GetExprHelper::Get(const parser::AllocateObject &x) {
  CheckMissingAnalysis(crashIfNoExpr_ && !x.typedExpr, context_, x);
  return x.typedExpr ? common::GetPtrFromOptional(x.typedExpr->v) : nullptr;
}
const SomeExpr *GetExprHelper::Get(const parser::PointerObject &x) {
  CheckMissingAnalysis(crashIfNoExpr_ && !x.typedExpr, context_, x);
  return x.typedExpr ? common::GetPtrFromOptional(x.typedExpr->v) : nullptr;
}

const evaluate::Assignment *GetAssignment(const parser::AssignmentStmt &x) {
  return x.typedAssignment ? common::GetPtrFromOptional(x.typedAssignment->v)
                           : nullptr;
}
const evaluate::Assignment *GetAssignment(
    const parser::PointerAssignmentStmt &x) {
  return x.typedAssignment ? common::GetPtrFromOptional(x.typedAssignment->v)
                           : nullptr;
}

const Symbol *FindInterface(const Symbol &symbol) {
  return common::visit(
      common::visitors{
          [](const ProcEntityDetails &details) {
            const Symbol *interface { details.interface().symbol() };
            return interface ? FindInterface(*interface) : nullptr;
          },
          [](const ProcBindingDetails &details) {
            return FindInterface(details.symbol());
          },
          [&](const SubprogramDetails &) { return &symbol; },
          [](const UseDetails &details) {
            return FindInterface(details.symbol());
          },
          [](const HostAssocDetails &details) {
            return FindInterface(details.symbol());
          },
          [](const GenericDetails &details) {
            return details.specific() ? FindInterface(*details.specific())
                                      : nullptr;
          },
          [](const auto &) -> const Symbol * { return nullptr; },
      },
      symbol.details());
}

const Symbol *FindSubprogram(const Symbol &symbol) {
  return common::visit(
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
          [](const GenericDetails &details) {
            return details.specific() ? FindSubprogram(*details.specific())
                                      : nullptr;
          },
          [](const auto &) -> const Symbol * { return nullptr; },
      },
      symbol.details());
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

const EquivalenceSet *FindEquivalenceSet(const Symbol &symbol) {
  const Symbol &ultimate{symbol.GetUltimate()};
  for (const EquivalenceSet &set : ultimate.owner().equivalenceSets()) {
    for (const EquivalenceObject &object : set) {
      if (object.symbol == ultimate) {
        return &set;
      }
    }
  }
  return nullptr;
}

bool IsOrContainsEventOrLockComponent(const Symbol &original) {
  const Symbol &symbol{ResolveAssociations(original)};
  if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (const DeclTypeSpec * type{details->type()}) {
      if (const DerivedTypeSpec * derived{type->AsDerived()}) {
        return IsEventTypeOrLockType(derived) ||
            FindEventOrLockPotentialComponent(*derived);
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

bool HasDeclarationInitializer(const Symbol &symbol) {
  if (IsNamedConstant(symbol)) {
    return false;
  } else if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    return object->init().has_value();
  } else if (const auto *proc{symbol.detailsIf<ProcEntityDetails>()}) {
    return proc->init().has_value();
  } else {
    return false;
  }
}

bool IsInitialized(
    const Symbol &symbol, bool ignoreDataStatements, bool ignoreAllocatable) {
  if (!ignoreAllocatable && IsAllocatable(symbol)) {
    return true;
  } else if (!ignoreDataStatements && symbol.test(Symbol::Flag::InDataStmt)) {
    return true;
  } else if (HasDeclarationInitializer(symbol)) {
    return true;
  } else if (IsNamedConstant(symbol) || IsFunctionResult(symbol) ||
      IsPointer(symbol)) {
    return false;
  } else if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (!object->isDummy() && object->type()) {
      if (const auto *derived{object->type()->AsDerived()}) {
        return derived->HasDefaultInitialization(ignoreAllocatable);
      }
    }
  }
  return false;
}

bool IsDestructible(const Symbol &symbol, const Symbol *derivedTypeSymbol) {
  if (IsAllocatable(symbol) || IsAutomatic(symbol)) {
    return true;
  } else if (IsNamedConstant(symbol) || IsFunctionResult(symbol) ||
      IsPointer(symbol)) {
    return false;
  } else if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (!object->isDummy() && object->type()) {
      if (const auto *derived{object->type()->AsDerived()}) {
        return &derived->typeSymbol() != derivedTypeSymbol &&
            derived->HasDestruction();
      }
    }
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

bool IsFinalizable(
    const Symbol &symbol, std::set<const DerivedTypeSpec *> *inProgress) {
  if (IsPointer(symbol)) {
    return false;
  }
  if (const auto *object{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (object->isDummy() && !IsIntentOut(symbol)) {
      return false;
    }
    const DeclTypeSpec *type{object->type()};
    const DerivedTypeSpec *typeSpec{type ? type->AsDerived() : nullptr};
    return typeSpec && IsFinalizable(*typeSpec, inProgress);
  }
  return false;
}

bool IsFinalizable(const DerivedTypeSpec &derived,
    std::set<const DerivedTypeSpec *> *inProgress) {
  if (!derived.typeSymbol().get<DerivedTypeDetails>().finals().empty()) {
    return true;
  }
  std::set<const DerivedTypeSpec *> basis;
  if (inProgress) {
    if (inProgress->find(&derived) != inProgress->end()) {
      return false; // don't loop on recursive type
    }
  } else {
    inProgress = &basis;
  }
  auto iterator{inProgress->insert(&derived).first};
  PotentialComponentIterator components{derived};
  bool result{bool{std::find_if(
      components.begin(), components.end(), [=](const Symbol &component) {
        return IsFinalizable(component, inProgress);
      })}};
  inProgress->erase(iterator);
  return result;
}

bool HasImpureFinal(const DerivedTypeSpec &derived) {
  if (const auto *details{
          derived.typeSymbol().detailsIf<DerivedTypeDetails>()}) {
    const auto &finals{details->finals()};
    return std::any_of(finals.begin(), finals.end(),
        [](const auto &x) { return !IsPureProcedure(*x.second); });
  } else {
    return false;
  }
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
  return ClassifyProcedure(symbol) == ProcedureDefinitionClass::External;
}

// Most scopes have no EQUIVALENCE, and this function is a fast no-op for them.
std::list<std::list<SymbolRef>> GetStorageAssociations(const Scope &scope) {
  UnorderedSymbolSet distinct;
  for (const EquivalenceSet &set : scope.equivalenceSets()) {
    for (const EquivalenceObject &object : set) {
      distinct.emplace(object.symbol);
    }
  }
  // This set is ordered by ascending offsets, with ties broken by greatest
  // size.  A multiset is used here because multiple symbols may have the
  // same offset and size; the symbols in the set, however, are distinct.
  std::multiset<SymbolRef, SymbolOffsetCompare> associated;
  for (SymbolRef ref : distinct) {
    associated.emplace(*ref);
  }
  std::list<std::list<SymbolRef>> result;
  std::size_t limit{0};
  const Symbol *currentCommon{nullptr};
  for (const Symbol &symbol : associated) {
    const Symbol *thisCommon{FindCommonBlockContaining(symbol)};
    if (result.empty() || symbol.offset() >= limit ||
        thisCommon != currentCommon) {
      // Start a new group
      result.emplace_back(std::list<SymbolRef>{});
      limit = 0;
      currentCommon = thisCommon;
    }
    result.back().emplace_back(symbol);
    limit = std::max(limit, symbol.offset() + symbol.size());
  }
  return result;
}

bool IsModuleProcedure(const Symbol &symbol) {
  return ClassifyProcedure(symbol) == ProcedureDefinitionClass::Module;
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

bool IsOrContainsPolymorphicComponent(const Symbol &original) {
  const Symbol &symbol{ResolveAssociations(original)};
  if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
    if (const DeclTypeSpec * type{details->type()}) {
      if (type->IsPolymorphic()) {
        return true;
      }
      if (const DerivedTypeSpec * derived{type->AsDerived()}) {
        return (bool)FindPolymorphicPotentialComponent(*derived);
      }
    }
  }
  return false;
}

class ImageControlStmtHelper {
  using ImageControlStmts =
      std::variant<parser::ChangeTeamConstruct, parser::CriticalConstruct,
          parser::EventPostStmt, parser::EventWaitStmt, parser::FormTeamStmt,
          parser::LockStmt, parser::SyncAllStmt, parser::SyncImagesStmt,
          parser::SyncMemoryStmt, parser::SyncTeamStmt, parser::UnlockStmt>;

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
  bool operator()(const parser::StopStmt &stmt) {
    // STOP is an image control statement; ERROR STOP is not
    return std::get<parser::StopStmt::Kind>(stmt.t) ==
        parser::StopStmt::Kind::Stop;
  }
  bool operator()(const parser::Statement<parser::ActionStmt> &stmt) {
    return common::visit(*this, stmt.statement.u);
  }

private:
  bool IsCoarrayObject(const parser::AllocateObject &allocateObject) {
    const parser::Name &name{GetLastName(allocateObject)};
    return name.symbol && evaluate::IsCoarray(*name.symbol);
  }
};

bool IsImageControlStmt(const parser::ExecutableConstruct &construct) {
  return common::visit(ImageControlStmtHelper{}, construct.u);
}

std::optional<parser::MessageFixedText> GetImageControlStmtCoarrayMsg(
    const parser::ExecutableConstruct &construct) {
  if (const auto *actionStmt{
          std::get_if<parser::Statement<parser::ActionStmt>>(&construct.u)}) {
    return common::visit(
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
  return common::visit(
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
  if (const auto *expr{GetExpr(nullptr, expression)}) {
    for (const Symbol &symbol : evaluate::CollectSymbols(*expr)) {
      if (evaluate::IsCoarray(symbol)) {
        return true;
      }
    }
  }
  return false;
}

bool IsAssumedType(const Symbol &symbol) {
  if (const DeclTypeSpec * type{symbol.GetType()}) {
    return type->IsAssumedType();
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
    if (FindModuleFileContaining(scope)) {
      // Don't enforce component accessibility checks in module files;
      // there may be forward-substituted named constants of derived type
      // whose structure constructors reference private components.
    } else if (const Scope *
        moduleScope{FindModuleContaining(symbol.owner())}) {
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

const DeclTypeSpec &FindOrInstantiateDerivedType(
    Scope &scope, DerivedTypeSpec &&spec, DeclTypeSpec::Category category) {
  spec.EvaluateParameters(scope.context());
  if (const DeclTypeSpec *
      type{scope.FindInstantiatedDerivedType(spec, category)}) {
    return *type;
  }
  // Create a new instantiation of this parameterized derived type
  // for this particular distinct set of actual parameter values.
  DeclTypeSpec &type{scope.MakeDerivedType(category, std::move(spec))};
  type.derivedTypeSpec().Instantiate(scope);
  return type;
}

const Symbol *FindSeparateModuleSubprogramInterface(const Symbol *proc) {
  if (proc) {
    if (const auto *subprogram{proc->detailsIf<SubprogramDetails>()}) {
      if (const Symbol * iface{subprogram->moduleInterface()}) {
        return iface;
      }
    }
  }
  return nullptr;
}

ProcedureDefinitionClass ClassifyProcedure(const Symbol &symbol) { // 15.2.2
  const Symbol &ultimate{symbol.GetUltimate()};
  if (!IsProcedure(ultimate)) {
    return ProcedureDefinitionClass::None;
  } else if (ultimate.attrs().test(Attr::INTRINSIC)) {
    return ProcedureDefinitionClass::Intrinsic;
  } else if (ultimate.attrs().test(Attr::EXTERNAL)) {
    return ProcedureDefinitionClass::External;
  } else if (const auto *procDetails{ultimate.detailsIf<ProcEntityDetails>()}) {
    if (procDetails->isDummy()) {
      return ProcedureDefinitionClass::Dummy;
    } else if (IsPointer(ultimate)) {
      return ProcedureDefinitionClass::Pointer;
    }
  } else if (const auto *nameDetails{
                 ultimate.detailsIf<SubprogramNameDetails>()}) {
    switch (nameDetails->kind()) {
    case SubprogramKind::Module:
      return ProcedureDefinitionClass::Module;
    case SubprogramKind::Internal:
      return ProcedureDefinitionClass::Internal;
    }
  } else if (const Symbol * subp{FindSubprogram(symbol)}) {
    if (const auto *subpDetails{subp->detailsIf<SubprogramDetails>()}) {
      if (subpDetails->stmtFunction()) {
        return ProcedureDefinitionClass::StatementFunction;
      }
    }
    switch (ultimate.owner().kind()) {
    case Scope::Kind::Global:
    case Scope::Kind::IntrinsicModules:
      return ProcedureDefinitionClass::External;
    case Scope::Kind::Module:
      return ProcedureDefinitionClass::Module;
    case Scope::Kind::MainProgram:
    case Scope::Kind::Subprogram:
      return ProcedureDefinitionClass::Internal;
    default:
      break;
    }
  }
  return ProcedureDefinitionClass::None;
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
  std::string designator;
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
  return std::find_if(ultimates.begin(), ultimates.end(),
      [](const Symbol &symbol) { return evaluate::IsCoarray(symbol); });
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

DirectComponentIterator::const_iterator FindAllocatableOrPointerDirectComponent(
    const DerivedTypeSpec &derived) {
  DirectComponentIterator directs{derived};
  return std::find_if(directs.begin(), directs.end(), IsAllocatableOrPointer);
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
    return IsPolymorphicAllocatable(x) && !evaluate::IsCoarray(x);
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

const Symbol *IsFunctionResultWithSameNameAsFunction(const Symbol &symbol) {
  if (IsFunctionResult(symbol)) {
    if (const Symbol * function{symbol.owner().symbol()}) {
      if (symbol.name() == function->name()) {
        return function;
      }
    }
    // Check ENTRY result symbols too
    const Scope &outer{symbol.owner().parent()};
    auto iter{outer.find(symbol.name())};
    if (iter != outer.end()) {
      const Symbol &outerSym{*iter->second};
      if (const auto *subp{outerSym.detailsIf<SubprogramDetails>()}) {
        if (subp->entryScope() == &symbol.owner() &&
            symbol.name() == outerSym.name()) {
          return &outerSym;
        }
      }
    }
  }
  return nullptr;
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

bool HasAlternateReturns(const Symbol &subprogram) {
  for (const auto *dummyArg : subprogram.get<SubprogramDetails>().dummyArgs()) {
    if (!dummyArg) {
      return true;
    }
  }
  return false;
}

bool InCommonBlock(const Symbol &symbol) {
  const auto *details{symbol.detailsIf<ObjectEntityDetails>()};
  return details && details->commonBlock();
}

const std::optional<parser::Name> &MaybeGetNodeName(
    const ConstructNode &construct) {
  return common::visit(
      common::visitors{
          [&](const parser::BlockConstruct *blockConstruct)
              -> const std::optional<parser::Name> & {
            return std::get<0>(blockConstruct->t).statement.v;
          },
          [&](const auto *a) -> const std::optional<parser::Name> & {
            return std::get<0>(std::get<0>(a->t).statement.t);
          },
      },
      construct);
}

std::optional<ArraySpec> ToArraySpec(
    evaluate::FoldingContext &context, const evaluate::Shape &shape) {
  if (auto extents{evaluate::AsConstantExtents(context, shape)}) {
    ArraySpec result;
    for (const auto &extent : *extents) {
      result.emplace_back(ShapeSpec::MakeExplicit(Bound{extent}));
    }
    return {std::move(result)};
  } else {
    return std::nullopt;
  }
}

std::optional<ArraySpec> ToArraySpec(evaluate::FoldingContext &context,
    const std::optional<evaluate::Shape> &shape) {
  return shape ? ToArraySpec(context, *shape) : std::nullopt;
}

bool HasDefinedIo(GenericKind::DefinedIo which, const DerivedTypeSpec &derived,
    const Scope *scope) {
  if (const Scope * dtScope{derived.scope()}) {
    for (const auto &pair : *dtScope) {
      const Symbol &symbol{*pair.second};
      if (const auto *generic{symbol.detailsIf<GenericDetails>()}) {
        GenericKind kind{generic->kind()};
        if (const auto *io{std::get_if<GenericKind::DefinedIo>(&kind.u)}) {
          if (*io == which) {
            return true; // type-bound GENERIC exists
          }
        }
      }
    }
  }
  if (scope) {
    SourceName name{GenericKind::AsFortran(which)};
    evaluate::DynamicType dyDerived{derived};
    for (; scope && !scope->IsGlobal(); scope = &scope->parent()) {
      auto iter{scope->find(name)};
      if (iter != scope->end()) {
        const auto &generic{iter->second->GetUltimate().get<GenericDetails>()};
        for (auto ref : generic.specificProcs()) {
          const Symbol &procSym{ref->GetUltimate()};
          if (const auto *subp{procSym.detailsIf<SubprogramDetails>()}) {
            if (!subp->dummyArgs().empty()) {
              if (const Symbol * first{subp->dummyArgs().at(0)}) {
                if (const DeclTypeSpec * dtSpec{first->GetType()}) {
                  if (auto dyDummy{evaluate::DynamicType::From(*dtSpec)}) {
                    if (dyDummy->IsTkCompatibleWith(dyDerived)) {
                      return true; // GENERIC or INTERFACE not in type
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
  return false;
}

const Symbol *FindUnsafeIoDirectComponent(GenericKind::DefinedIo which,
    const DerivedTypeSpec &derived, const Scope *scope) {
  if (HasDefinedIo(which, derived, scope)) {
    return nullptr;
  }
  if (const Scope * dtScope{derived.scope()}) {
    for (const auto &pair : *dtScope) {
      const Symbol &symbol{*pair.second};
      if (IsAllocatableOrPointer(symbol)) {
        return &symbol;
      }
      if (const auto *details{symbol.detailsIf<ObjectEntityDetails>()}) {
        if (const DeclTypeSpec * type{details->type()}) {
          if (type->category() == DeclTypeSpec::Category::TypeDerived) {
            if (const Symbol *
                bad{FindUnsafeIoDirectComponent(
                    which, type->derivedTypeSpec(), scope)}) {
              return bad;
            }
          }
        }
      }
    }
  }
  return nullptr;
}

} // namespace Fortran::semantics
