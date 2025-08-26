//===--- SubprogramTrampolineCheck.cpp - flang-tidy -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SubprogramTrampolineCheck.h"
#include "../utils/CollectActualArguments.h"
#include "flang/Evaluate/call.h"
#include "flang/Evaluate/tools.h"
#include "flang/Evaluate/variable.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/symbol.h"
#include "flang/Semantics/tools.h"
#include "flang/Semantics/type.h"

namespace Fortran::tidy::bugprone {

using namespace parser::literals;
void SubprogramTrampolineCheck::Enter(const parser::CallStmt &callStmt) {
  const auto *procedureRef = callStmt.typedCall.get();
  if (procedureRef) {
    for (const auto &arg : procedureRef->arguments()) {
      if (!arg)
        continue;
      if (const semantics::SomeExpr *argExpr{arg->UnwrapExpr()}) {
        if (!evaluate::IsProcedureDesignator(*argExpr))
          continue;
        const auto proc = std::get<evaluate::ProcedureDesignator>(argExpr->u);
        if (const auto *symbol{proc.GetSymbol()}) {
          if (symbol->has<semantics::SubprogramDetails>()) {
            const auto &owner = symbol->owner();
            if (owner.IsModule())
              continue;

            Say(callStmt.source,
                "contained subprogram '%s' is passed as an argument"_warn_en_US,
                symbol->name().ToString());
          }
        }
      }
    }
  }
}

void SubprogramTrampolineCheck::Enter(const parser::Expr &e) {
  const auto *expr = semantics::GetExpr(context()->getSemanticsContext(), e);
  if (!expr) {
    return;
  }

  if (std::holds_alternative<common::Indirection<parser::FunctionReference>>(
          e.u)) {
    evaluate::ActualArgumentSet argSet{evaluate::CollectActualArguments(*expr)};
    for (const evaluate::ActualArgumentRef &argRef : argSet) {
      if (const semantics::SomeExpr *argExpr{argRef->UnwrapExpr()}) {
        if (!evaluate::IsProcedureDesignator(*argExpr))
          continue;
        const auto proc = std::get<evaluate::ProcedureDesignator>(argExpr->u);
        if (const auto *symbol{proc.GetSymbol()}) {
          if (symbol->has<semantics::SubprogramDetails>()) {
            const auto &owner = symbol->owner();
            if (owner.IsModule())
              continue;
            Say(e.source,
                "contained subprogram '%s' is passed as an argument"_warn_en_US,
                symbol->name().ToString());
          }
        }
      }
    }
  }
}

} // namespace Fortran::tidy::bugprone
