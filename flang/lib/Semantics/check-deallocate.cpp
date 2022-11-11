//===-- lib/Semantics/check-deallocate.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-deallocate.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {

void DeallocateChecker::Leave(const parser::DeallocateStmt &deallocateStmt) {
  for (const parser::AllocateObject &allocateObject :
      std::get<std::list<parser::AllocateObject>>(deallocateStmt.t)) {
    common::visit(
        common::visitors{
            [&](const parser::Name &name) {
              auto const *symbol{name.symbol};
              if (context_.HasError(symbol)) {
                // already reported an error
              } else if (!IsVariableName(*symbol)) {
                context_.Say(name.source,
                    "name in DEALLOCATE statement must be a variable name"_err_en_US);
              } else if (!IsAllocatableOrPointer(
                             symbol->GetUltimate())) { // C932
                context_.Say(name.source,
                    "name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute"_err_en_US);
              } else {
                context_.CheckIndexVarRedefine(name);
              }
            },
            [&](const parser::StructureComponent &structureComponent) {
              // Only perform structureComponent checks it was successfully
              // analyzed in expression analysis.
              if (GetExpr(context_, allocateObject)) {
                if (!IsAllocatableOrPointer(
                        *structureComponent.component.symbol)) { // C932
                  context_.Say(structureComponent.component.source,
                      "component in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute"_err_en_US);
                }
              }
            },
        },
        allocateObject.u);
  }
  bool gotStat{false}, gotMsg{false};
  for (const parser::StatOrErrmsg &deallocOpt :
      std::get<std::list<parser::StatOrErrmsg>>(deallocateStmt.t)) {
    common::visit(
        common::visitors{
            [&](const parser::StatVariable &) {
              if (gotStat) {
                context_.Say(
                    "STAT may not be duplicated in a DEALLOCATE statement"_err_en_US);
              }
              gotStat = true;
            },
            [&](const parser::MsgVariable &) {
              if (gotMsg) {
                context_.Say(
                    "ERRMSG may not be duplicated in a DEALLOCATE statement"_err_en_US);
              }
              gotMsg = true;
            },
        },
        deallocOpt.u);
  }
}
} // namespace Fortran::semantics
