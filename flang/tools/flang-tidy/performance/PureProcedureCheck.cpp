//===--- PureProcedureCheck.cpp - flang-tidy ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PureProcedureCheck.h"
#include "../utils/SymbolUtils.h"
#include "flang/Evaluate/tools.h"
#include "flang/Semantics/scope.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"

namespace Fortran::tidy::performance {

static void PopulateProcedures(
    const semantics::Scope &scope,
    std::unordered_map<const semantics::Symbol *, bool> &pureProcedures) {
  if (scope.IsModuleFile())
    return;

  for (const auto &pair : scope) {
    const semantics::Symbol &symbol{*pair.second};
    const auto &ultimate = symbol.GetUltimate();
    const auto *subprogramDetails =
        ultimate.detailsIf<semantics::SubprogramDetails>();
    if (semantics::IsProcedure(symbol) && !utils::IsFromModFileSafe(ultimate) &&
        (subprogramDetails && !subprogramDetails->isInterface())) {
      pureProcedures[&ultimate] = true;
    }
  }

  for (const auto &child : scope.children()) {
    PopulateProcedures(child, pureProcedures);
  }
}

static bool CheckSymbolIsPure(const semantics::Symbol &symbol,
                              const semantics::Scope &scope) {
  if (scope.symbol()) {
    if (const auto &details =
            scope.symbol()->detailsIf<semantics::SubprogramDetails>();
        !(details && details->isInterface()) &&
        !semantics::FindCommonBlockContaining(symbol) && IsSaved(symbol)) {
      return false;
    }
  }

  if (symbol.attrs().test(semantics::Attr::VOLATILE) && IsDummy(symbol)) {
    return false;
  }

  if (IsProcedure(symbol) && !IsPureProcedure(symbol) && IsDummy(symbol)) {
    return false;
  }

  return true;
}

static void CheckPureSymbols(
    const semantics::Scope &scope,
    std::unordered_map<const semantics::Symbol *, bool> &pureProcedures) {

  if (scope.IsModuleFile())
    return;

  if (!scope.IsTopLevel()) {
    for (const auto &pair : scope) {
      const semantics::Symbol &symbol{*pair.second};
      const semantics::Scope &scope2{symbol.owner()};
      const semantics::Scope &unit{GetProgramUnitContaining(scope2)};
      // unit should not be an interface
      if (!CheckSymbolIsPure(symbol, scope) &&
          unit.symbol() != &symbol.GetUltimate()) {
        pureProcedures[&unit.symbol()->GetUltimate()] = false;
      }
    }
  }
  for (const auto &child : scope.children()) {
    CheckPureSymbols(child, pureProcedures);
  }
}

static std::unordered_map<const semantics::Symbol *, const semantics::Symbol *>
    procBindingDetailsSymbolsMap;

static void MakeProcBindingSymbolSet(semantics::SemanticsContext &context,
                                     const semantics::Scope &scope) {
  for (const auto &pair : scope) {
    const semantics::Symbol &symbol = *pair.second;
    if (auto *details{symbol.detailsIf<semantics::ProcBindingDetails>()}) {
      procBindingDetailsSymbolsMap[&details->symbol().GetUltimate()] = &symbol;
    }
  }

  for (const semantics::Scope &child : scope.children()) {
    MakeProcBindingSymbolSet(context, child);
  }
}

PureProcedureCheck::PureProcedureCheck(llvm::StringRef name,
                                       FlangTidyContext *context)
    : FlangTidyCheck{name, context} {
  MakeProcBindingSymbolSet(context->getSemanticsContext(),
                           context->getSemanticsContext().globalScope());

  PopulateProcedures(context->getSemanticsContext().globalScope(),
                     pureProcedures_);
  CheckPureSymbols(context->getSemanticsContext().globalScope(),
                   pureProcedures_);
}

using namespace parser::literals;
void PureProcedureCheck::SetImpure() {
  const auto location{context()->getSemanticsContext().location()};
  if (!location) {
    return;
  }

  const semantics::Scope &scope{
      context()->getSemanticsContext().FindScope(*location)};

  if (scope.IsTopLevel())
    return;

  const semantics::Scope &unit{GetProgramUnitContaining(scope)};
  const auto &ultimateSymbol = unit.symbol()->GetUltimate();

  if (semantics::IsProcedure(unit)) {
    pureProcedures_[&ultimateSymbol] = false;
  }

  // propagate impurity to other procedures
  for (const auto &pair : procBindingDetailsSymbolsMap) {
    if (pair.first == &ultimateSymbol) {
      pureProcedures_[pair.second] = false;
    }
  }
}

// C1596: external I/O
void PureProcedureCheck::Leave(const parser::BackspaceStmt &) { SetImpure(); }
void PureProcedureCheck::Leave(const parser::CloseStmt &) { SetImpure(); }
void PureProcedureCheck::Leave(const parser::EndfileStmt &) { SetImpure(); }
void PureProcedureCheck::Leave(const parser::FlushStmt &) { SetImpure(); }
void PureProcedureCheck::Leave(const parser::InquireStmt &) { SetImpure(); }
void PureProcedureCheck::Leave(const parser::OpenStmt &) { SetImpure(); }
void PureProcedureCheck::Leave(const parser::PrintStmt &) { SetImpure(); }
void PureProcedureCheck::Leave(const parser::RewindStmt &) { SetImpure(); }
void PureProcedureCheck::Leave(const parser::WaitStmt &) { SetImpure(); }
// C1597: read/write
void PureProcedureCheck::Leave(const parser::ReadStmt &) { SetImpure(); }
void PureProcedureCheck::Leave(const parser::WriteStmt &) { SetImpure(); }

// assignment
static bool IsPointerDummyOfPureFunction(const semantics::Symbol &x) {
  return IsPointerDummy(x) && FindPureProcedureContaining(x.owner()) &&
         x.owner().symbol() && IsFunction(*x.owner().symbol());
}

static const char *WhyBaseObjectIsSuspicious(const semantics::Symbol &x,
                                             const semantics::Scope &scope) {
  if (IsHostAssociatedIntoSubprogram(x, scope)) {
    return "host-associated";
  }
  if (IsUseAssociated(x, scope)) {
    return "USE-associated";
  }
  if (IsPointerDummyOfPureFunction(x)) {
    return "a POINTER dummy argument of a pure function";
  }
  if (IsIntentIn(x)) {
    return "an INTENT(IN) dummy argument";
  }
  if (FindCommonBlockContaining(x)) {
    return "in a COMMON block";
  }
  return nullptr;
}

static std::optional<std::string>
GetPointerComponentDesignatorName(const semantics::SomeExpr &expr) {
  if (const auto *derived{
          evaluate::GetDerivedTypeSpec(evaluate::DynamicType::From(expr))}) {
    semantics::PotentialAndPointerComponentIterator potentials{*derived};
    if (auto pointer{std::find_if(potentials.begin(), potentials.end(),
                                  semantics::IsPointer)}) {
      return pointer.BuildResultDesignatorName();
    }
  }
  return std::nullopt;
}

// C1593 (4)
void PureProcedureCheck::Leave(const parser::AssignmentStmt &assignment) {
  // get the rhs
  const auto &rhs{std::get<parser::Expr>(assignment.t)};
  // get the evaluate::Expr
  const auto *expr{semantics::GetExpr(context()->getSemanticsContext(), rhs)};

  const semantics::Scope &scope{
      context()->getSemanticsContext().FindScope(rhs.source)};

  if (const semantics::Symbol *base{GetFirstSymbol(expr)}) {
    if (const char *why{
            WhyBaseObjectIsSuspicious(base->GetUltimate(), scope)}) {
      if (auto pointer{GetPointerComponentDesignatorName(*expr)}) {
        // mark the procedure as impure
        (void)why;
        SetImpure();
      }
    }
  }
}

// C1592
void PureProcedureCheck::Leave(const parser::Name &n) {
  if (n.symbol && n.symbol->attrs().test(semantics::Attr::VOLATILE)) {
    SetImpure();
  }
}

// C1598: pure procs cant have image control statements
void PureProcedureCheck::Enter(const parser::ExecutableConstruct &exec) {
  if (semantics::IsImageControlStmt(exec)) {
    SetImpure();
  }
}

// check Call Stmt
void PureProcedureCheck::Enter(const parser::CallStmt &callStmt) {
  const auto *procedureRef = callStmt.typedCall.get();
  if (procedureRef) {
    const auto *symbol{procedureRef->proc().GetSymbol()};
    if (!symbol) {
      return;
    }

    // if the called function isnt pure, we cant be pure
    if (!semantics::IsPureProcedure(*symbol))
      SetImpure();
  }
}

void PureProcedureCheck::Leave(const parser::Program &program) {
  // tell us about all the procedure that could be pure, but arent
  for (const auto &pair : pureProcedures_) {
    // it should be pure, but isnt - and its not intrinsic, and not elemental,
    // and not an interface
    // check if the symbol is the Ultimate symbol
    const auto &symbol = pair.first->GetUltimate();
    // if (symbol != *pair.first) {
    //   continue;
    // }

    // make sure its not being mapped to from procBindingDetailsSymbolsMap
    bool cont = false;
    for (const auto &procBindingPair : procBindingDetailsSymbolsMap) {
      if (procBindingPair.second == pair.first) {
        cont = true;
        break;
      }
    }
    if (cont) {
      continue;
    }

    if (utils::IsFromModFileSafe(symbol)) {
      continue;
    }

    // TODO: skip interfaces
    if (pair.second && !pair.first->attrs().test(semantics::Attr::PURE) &&
        !pair.first->attrs().test(semantics::Attr::INTRINSIC) &&
        !pair.first->attrs().test(semantics::Attr::ELEMENTAL) &&
        !pair.first->attrs().test(semantics::Attr::ABSTRACT) &&
        !pair.first->attrs().test(semantics::Attr::EXTERNAL) &&
        !pair.first->IsFromModFile()
        // if its been derived somewhere, we dont care
        && procBindingDetailsSymbolsMap.find(pair.first) ==
               procBindingDetailsSymbolsMap.end()) {
      Say(pair.first->name(),
          "Procedure '%s' could be PURE but is not"_warn_en_US,
          pair.first->name());
    }
  }
}

} // namespace Fortran::tidy::performance
