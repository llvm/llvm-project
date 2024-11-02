//===-- lib/Semantics/check-deallocate.cpp --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "check-deallocate.h"
#include "flang/Evaluate/type.h"
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
              } else if (CheckPolymorphism(name.source, *symbol)) {
                context_.CheckIndexVarRedefine(name);
              }
            },
            [&](const parser::StructureComponent &structureComponent) {
              // Only perform structureComponent checks it was successfully
              // analyzed in expression analysis.
              if (GetExpr(context_, allocateObject)) {
                if (const Symbol *symbol{structureComponent.component.symbol}) {
                  if (!IsAllocatableOrPointer(*symbol)) { // C932
                    context_.Say(structureComponent.component.source,
                        "component in DEALLOCATE statement must have the ALLOCATABLE or POINTER attribute"_err_en_US);
                  } else {
                    CheckPolymorphism(
                        structureComponent.component.source, *symbol);
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

bool DeallocateChecker::CheckPolymorphism(
    parser::CharBlock source, const Symbol &symbol) {
  if (FindPureProcedureContaining(context_.FindScope(source))) {
    if (auto type{evaluate::DynamicType::From(symbol)}) {
      if (type->IsPolymorphic()) {
        context_.Say(source,
            "'%s' may not be deallocated in a pure procedure because it is polymorphic"_err_en_US,
            source);
        return false;
      }
      if (!type->IsUnlimitedPolymorphic() &&
          type->category() == TypeCategory::Derived) {
        if (auto iter{FindPolymorphicAllocatableUltimateComponent(
                type->GetDerivedTypeSpec())}) {
          context_.Say(source,
              "'%s' may not be deallocated in a pure procedure because its type has a polymorphic allocatable ultimate component '%s'"_err_en_US,
              source, iter->name());
          return false;
        }
      }
    }
  }
  return true;
}
} // namespace Fortran::semantics
