//===-- lib/Semantics/check-nullify.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-nullify.h"
#include "definable.h"
#include "flang/Evaluate/expression.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {

void NullifyChecker::Leave(const parser::NullifyStmt &nullifyStmt) {
  CHECK(context_.location());
  const Scope &scope{context_.FindScope(*context_.location())};
  for (const parser::PointerObject &pointerObject : nullifyStmt.v) {
    common::visit(
        common::visitors{
            [&](const parser::Name &name) {
              if (name.symbol) {
                if (auto whyNot{WhyNotDefinable(name.source, scope,
                        DefinabilityFlags{DefinabilityFlag::PointerDefinition},
                        *name.symbol)}) {
                  context_.messages()
                      .Say(name.source,
                          "'%s' may not appear in NULLIFY"_err_en_US,
                          name.source)
                      .Attach(std::move(*whyNot));
                }
              }
            },
            [&](const parser::StructureComponent &structureComponent) {
              const auto &component{structureComponent.component};
              SourceName at{component.source};
              if (const auto *checkedExpr{GetExpr(context_, pointerObject)}) {
                if (auto whyNot{WhyNotDefinable(at, scope,
                        DefinabilityFlags{DefinabilityFlag::PointerDefinition},
                        *checkedExpr)}) {
                  context_.messages()
                      .Say(at, "'%s' may not appear in NULLIFY"_err_en_US, at)
                      .Attach(std::move(*whyNot));
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
  // This restriction is the programmer's responsibility.
  // Some dependencies can be found compile time or at
  // runtime, but for now we choose to skip such checks.
}
} // namespace Fortran::semantics
