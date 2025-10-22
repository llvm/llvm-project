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
  bool gotStat{false}, gotMsg{false};
  const parser::Name *statVar{nullptr}, *msgVar{nullptr};
  for (const parser::StatOrErrmsg &deallocOpt :
      std::get<std::list<parser::StatOrErrmsg>>(deallocateStmt.t)) {
    common::visit(
        common::visitors{
            [&](const parser::StatVariable &var) {
              if (gotStat) {
                context_.Say(
                    "STAT may not be duplicated in a DEALLOCATE statement"_err_en_US);
              }
              if (const auto *designator{
                      parser::Unwrap<parser::Designator>(var)}) {
                statVar = &parser::GetLastName(*designator);
              }
              gotStat = true;
            },
            [&](const parser::MsgVariable &var) {
              WarnOnDeferredLengthCharacterScalar(context_,
                  GetExpr(context_, var),
                  parser::UnwrapRef<parser::Variable>(var).GetSource(),
                  "ERRMSG=");
              if (gotMsg) {
                context_.Say(
                    "ERRMSG may not be duplicated in a DEALLOCATE statement"_err_en_US);
              }
              if (const auto *designator{
                      parser::Unwrap<parser::Designator>(var)}) {
                msgVar = &parser::GetLastName(*designator);
              }
              gotMsg = true;
            },
        },
        deallocOpt.u);
  }
  for (const parser::AllocateObject &allocateObject :
      std::get<std::list<parser::AllocateObject>>(deallocateStmt.t)) {
    const Symbol *ultimate{nullptr};
    common::visit(
        common::visitors{
            [&](const parser::Name &name) {
              if (name.symbol) {
                ultimate = &name.symbol->GetUltimate();
              }
              if (context_.HasError(ultimate)) {
                // already reported an error
              } else if (!IsVariableName(*ultimate)) {
                context_.Say(name.source,
                    "Name in DEALLOCATE statement must be a variable name"_err_en_US);
              } else if (!IsAllocatableOrObjectPointer(ultimate)) { // C936
                context_.Say(name.source,
                    "Name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute"_err_en_US);
              } else if (auto whyNot{WhyNotDefinable(name.source,
                             context_.FindScope(name.source),
                             {DefinabilityFlag::PointerDefinition,
                                 DefinabilityFlag::AcceptAllocatable,
                                 DefinabilityFlag::PotentialDeallocation},
                             *ultimate)}) {
                // Catch problems with non-definability of the
                // pointer/allocatable
                context_
                    .Say(name.source,
                        "Name in DEALLOCATE statement is not definable"_err_en_US)
                    .Attach(std::move(
                        whyNot->set_severity(parser::Severity::Because)));
              } else if (auto whyNot{WhyNotDefinable(name.source,
                             context_.FindScope(name.source),
                             DefinabilityFlags{}, *ultimate)}) {
                // Catch problems with non-definability of the dynamic object
                context_
                    .Say(name.source,
                        "Object in DEALLOCATE statement is not deallocatable"_err_en_US)
                    .Attach(std::move(
                        whyNot->set_severity(parser::Severity::Because)));
              } else {
                context_.CheckIndexVarRedefine(name);
              }
            },
            [&](const parser::StructureComponent &structureComponent) {
              // Only perform structureComponent checks if it was successfully
              // analyzed by expression analysis.
              auto source{structureComponent.component.source};
              if (structureComponent.component.symbol) {
                ultimate = &structureComponent.component.symbol->GetUltimate();
              }
              if (const auto *expr{GetExpr(context_, allocateObject)}) {
                if (!IsAllocatableOrObjectPointer(ultimate)) { // F'2023 C936
                  context_.Say(source,
                      "Component in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute"_err_en_US);
                } else if (auto whyNot{WhyNotDefinable(source,
                               context_.FindScope(source),
                               {DefinabilityFlag::PointerDefinition,
                                   DefinabilityFlag::AcceptAllocatable,
                                   DefinabilityFlag::PotentialDeallocation},
                               *expr)}) {
                  context_
                      .Say(source,
                          "Name in DEALLOCATE statement is not definable"_err_en_US)
                      .Attach(std::move(
                          whyNot->set_severity(parser::Severity::Because)));
                } else if (auto whyNot{WhyNotDefinable(source,
                               context_.FindScope(source), DefinabilityFlags{},
                               *expr)}) {
                  context_
                      .Say(source,
                          "Object in DEALLOCATE statement is not deallocatable"_err_en_US)
                      .Attach(std::move(
                          whyNot->set_severity(parser::Severity::Because)));
                } else if (evaluate::ExtractCoarrayRef(*expr)) { // F'2023 C955
                  context_.Say(source,
                      "Component in DEALLOCATE statement may not be coindexed"_err_en_US);
                }
              }
            },
        },
        allocateObject.u);
    if (ultimate) {
      if (gotStat && statVar) {
        if (const Symbol *symbol{statVar->symbol};
            symbol && *ultimate == symbol->GetUltimate()) {
          context_.Say(statVar->source,
              "STAT variable in DEALLOCATE must not be the variable being deallocated"_err_en_US);
        }
      }
      if (gotMsg && msgVar) {
        if (const Symbol *symbol{msgVar->symbol};
            symbol && *ultimate == symbol->GetUltimate()) {
          context_.Say(msgVar->source,
              "ERRMSG variable in DEALLOCATE must not be the variable being deallocated"_err_en_US);
        }
      }
    }
  }
}

} // namespace Fortran::semantics
