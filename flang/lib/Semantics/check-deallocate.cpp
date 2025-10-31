//===-- lib/Semantics/check-deallocate.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-deallocate.h"
#include "check-allocate.h"
#include "definable.h"
#include "flang/Evaluate/type.h"
#include "flang/Parser/message.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/expression.h"
#include "flang/Semantics/tools.h"
#include <optional>

namespace Fortran::semantics {

void DeallocateChecker::Leave(const parser::DeallocateStmt &deallocateStmt) {
  bool gotStat{false}, gotMsg{false};
  const SomeExpr *statVar{nullptr}, *msgVar{nullptr};
  std::optional<parser::CharBlock> statSource;
  std::optional<parser::CharBlock> msgSource;
  for (const parser::StatOrErrmsg &deallocOpt :
      std::get<std::list<parser::StatOrErrmsg>>(deallocateStmt.t)) {
    common::visit(
        common::visitors{
            [&](const parser::StatVariable &var) {
              if (gotStat) {
                context_.Say(
                    "STAT may not be duplicated in a DEALLOCATE statement"_err_en_US);
              }
              gotStat = true;
              statVar = GetExpr(context_, var);
              statSource = parser::Unwrap<parser::Variable>(var)->GetSource();
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
              gotMsg = true;
              msgVar = GetExpr(context_, var);
              msgSource = parser::Unwrap<parser::Variable>(var)->GetSource();
            },
        },
        deallocOpt.u);
  }
  for (const parser::AllocateObject &allocateObject :
      std::get<std::list<parser::AllocateObject>>(deallocateStmt.t)) {
    parser::CharBlock source;
    common::visit(
        common::visitors{
            [&](const parser::Name &name) {
              const Symbol *symbol{
                  name.symbol ? &name.symbol->GetUltimate() : nullptr};
              source = name.source;
              if (context_.HasError(symbol)) {
                // already reported an error
              } else if (!IsVariableName(*symbol)) {
                context_.Say(source,
                    "Name in DEALLOCATE statement must be a variable name"_err_en_US);
              } else if (!IsAllocatableOrObjectPointer(symbol)) { // C936
                context_.Say(source,
                    "Name in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute"_err_en_US);
              } else if (auto whyNot{
                             WhyNotDefinable(source, context_.FindScope(source),
                                 {DefinabilityFlag::PointerDefinition,
                                     DefinabilityFlag::AcceptAllocatable,
                                     DefinabilityFlag::PotentialDeallocation},
                                 *symbol)}) {
                // Catch problems with non-definability of the
                // pointer/allocatable
                context_
                    .Say(source,
                        "Name in DEALLOCATE statement is not definable"_err_en_US)
                    .Attach(std::move(
                        whyNot->set_severity(parser::Severity::Because)));
              } else if (auto whyNot{
                             WhyNotDefinable(source, context_.FindScope(source),
                                 DefinabilityFlags{}, *symbol)}) {
                // Catch problems with non-definability of the dynamic object
                context_
                    .Say(source,
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
              source = structureComponent.component.source;
              if (const auto *expr{GetExpr(context_, allocateObject)}) {
                if (const Symbol *symbol{structureComponent.component.symbol
                            ? &structureComponent.component.symbol
                                  ->GetUltimate()
                            : nullptr};
                    !IsAllocatableOrObjectPointer(symbol)) { // F'2023 C936
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
    if (const SomeExpr *allocObj{GetExpr(context_, allocateObject)}) {
      if (AreSameAllocation(allocObj, statVar)) {
        context_.Say(statSource.value_or(source),
            "STAT variable in DEALLOCATE must not be the variable being deallocated"_err_en_US);
      }
      if (AreSameAllocation(allocObj, msgVar)) {
        context_.Say(msgSource.value_or(source),
            "ERRMSG variable in DEALLOCATE must not be the variable being deallocated"_err_en_US);
      }
    }
  }
}

} // namespace Fortran::semantics
