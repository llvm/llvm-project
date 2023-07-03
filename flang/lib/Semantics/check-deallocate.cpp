//===-- lib/Semantics/check-deallocate.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-deallocate.h"
#include "definable.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/tools.h"

namespace Fortran::semantics {

void DeallocateChecker::Leave(const parser::DeallocateStmt &deallocateStmt) {
  for (const parser::AllocateObject &allocateObject :
      std::get<std::list<parser::AllocateObject>>(deallocateStmt.t)) {
    parser::CharBlock source;
    const Symbol *symbol{nullptr};
    common::visit(
        common::visitors{
            [&](const parser::Name &name) {
              source = name.source;
              symbol = name.symbol;
              if (context_.HasError(symbol)) {
                // already reported an error
              } else if (!IsVariableName(*symbol)) {
                context_.Say(name.source,
                    "Name in DEALLOCATE statement must be a variable name"_err_en_US);
              } else if (!IsAllocatableOrPointer(
                             symbol->GetUltimate())) { // C932
                context_.Say(name.source,
                    "Name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute"_err_en_US);
              } else if (auto whyNot{WhyNotDefinable(name.source,
                             context_.FindScope(name.source),
                             {DefinabilityFlag::PointerDefinition,
                                 DefinabilityFlag::AcceptAllocatable},
                             *symbol)}) {
                // Catch problems with non-definability of the
                // pointer/allocatable
                context_
                    .Say(name.source,
                        "Name in DEALLOCATE statement is not definable"_err_en_US)
                    .Attach(std::move(*whyNot));
              } else if (auto whyNot{WhyNotDefinable(name.source,
                             context_.FindScope(name.source),
                             DefinabilityFlags{}, *symbol)}) {
                // Catch problems with non-definability of the dynamic object
                context_
                    .Say(name.source,
                        "Object in DEALLOCATE statement is not deallocatable"_err_en_US)
                    .Attach(std::move(*whyNot));
              } else {
                context_.CheckIndexVarRedefine(name);
              }
            },
            [&](const parser::StructureComponent &structureComponent) {
              // Only perform structureComponent checks if it was successfully
              // analyzed by expression analysis.
              source = structureComponent.component.source;
              symbol = structureComponent.component.symbol;
              if (const auto *expr{GetExpr(context_, allocateObject)}) {
                if (symbol) {
                  if (!IsAllocatableOrPointer(*symbol)) { // C932
                    context_.Say(source,
                        "Component in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute"_err_en_US);
                  } else if (auto whyNot{WhyNotDefinable(source,
                                 context_.FindScope(source),
                                 {DefinabilityFlag::PointerDefinition,
                                     DefinabilityFlag::AcceptAllocatable},
                                 *expr)}) {
                    context_
                        .Say(source,
                            "Name in DEALLOCATE statement is not definable"_err_en_US)
                        .Attach(std::move(*whyNot));
                  } else if (auto whyNot{WhyNotDefinable(source,
                                 context_.FindScope(source),
                                 DefinabilityFlags{}, *expr)}) {
                    context_
                        .Say(source,
                            "Object in DEALLOCATE statement is not deallocatable"_err_en_US)
                        .Attach(std::move(*whyNot));
                  }
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
            [&](const parser::MsgVariable &var) {
              WarnOnDeferredLengthCharacterScalar(context_,
                  GetExpr(context_, var), var.v.thing.thing.GetSource(),
                  "ERRMSG=");
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
