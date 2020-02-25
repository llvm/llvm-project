//===-- lib/Semantics/check-data.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-data.h"

namespace Fortran::semantics {

void DataChecker::Leave(const parser::DataStmtConstant &dataConst) {
  if (auto *structure{
          std::get_if<parser::StructureConstructor>(&dataConst.u)}) {
    for (const auto &component :
        std::get<std::list<parser::ComponentSpec>>(structure->t)) {
      const parser::Expr &parsedExpr{
          std::get<parser::ComponentDataSource>(component.t).v.value()};
      if (const auto *expr{GetExpr(parsedExpr)}) {
        if (!evaluate::IsConstantExpr(*expr)) {  // C884
          context_.Say(parsedExpr.source,
              "Structure constructor in data value must be a constant expression"_err_en_US);
        }
      }
    }
  }
  // TODO: C886 and C887 for data-stmt-constant
}

// TODO: C874-C881

void DataChecker::Leave(const parser::DataStmtRepeat &dataRepeat) {
  if (const auto *designator{parser::Unwrap<parser::Designator>(dataRepeat)}) {
    if (auto *dataRef{std::get_if<parser::DataRef>(&designator->u)}) {
      evaluate::ExpressionAnalyzer exprAnalyzer{context_};
      if (MaybeExpr checked{exprAnalyzer.Analyze(*dataRef)}) {
        auto expr{
            evaluate::Fold(context_.foldingContext(), std::move(checked))};
        if (auto i64{ToInt64(expr)}) {
          if (*i64 < 0) {  // C882
            context_.Say(designator->source,
                "Repeat count for data value must not be negative"_err_en_US);
          }
        }
      }
    }
  }
}
}
