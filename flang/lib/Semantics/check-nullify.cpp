//===-- lib/Semantics/check-nullify.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-nullify.h"
#include "assignment.h"
#include "flang/Evaluate/expression.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {

void NullifyChecker::Leave(const parser::NullifyStmt &nullifyStmt) {
  CHECK(context_.location());
  const Scope &scope{context_.FindScope(*context_.location())};
  const Scope *pure{FindPureProcedureContaining(scope)};
  parser::ContextualMessages messages{
      *context_.location(), &context_.messages()};
  for (const parser::PointerObject &pointerObject : nullifyStmt.v) {
    std::visit(
        common::visitors{
            [&](const parser::Name &name) {
              const Symbol &symbol{DEREF(name.symbol)};
              if (context_.HasError(&symbol)) {
                // already reported an error
              } else if (!IsVariableName(symbol) && !IsProcName(symbol)) {
                messages.Say(name.source,
                    "name in NULLIFY statement must be a variable or procedure pointer name"_err_en_US);
              } else if (!IsPointer(symbol)) { // C951
                messages.Say(name.source,
                    "name in NULLIFY statement must have the POINTER attribute"_err_en_US);
              } else if (pure) {
                CheckDefinabilityInPureScope(messages, symbol, scope, *pure);
              }
            },
            [&](const parser::StructureComponent &structureComponent) {
              evaluate::ExpressionAnalyzer analyzer{context_};
              if (MaybeExpr checked{analyzer.Analyze(structureComponent)}) {
                if (!IsPointer(*structureComponent.component.symbol)) { // C951
                  messages.Say(structureComponent.component.source,
                      "component in NULLIFY statement must have the POINTER attribute"_err_en_US);
                } else if (pure) {
                  if (const Symbol * symbol{GetFirstSymbol(checked)}) {
                    CheckDefinabilityInPureScope(
                        messages, *symbol, scope, *pure);
                  }
                }
              }
            },
        },
        pointerObject.u);
  }
  // From 9.7.3.1(1)
  //   A pointer-object shall not depend on the value,
  //   bounds, or association status of another pointer-
  //   object in the same NULLIFY statement.
  // This restriction is the programmer's responsibilty.
  // Some dependencies can be found compile time or at
  // runtime, but for now we choose to skip such checks.
}
} // namespace Fortran::semantics
