//===-- flang/Parser/openmp-utils.cpp -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Common OpenMP utilities.
//
//===----------------------------------------------------------------------===//

#include "flang/Parser/openmp-utils.h"

#include "flang/Common/indirection.h"
#include "flang/Common/template.h"
#include "flang/Common/visit.h"

#include <tuple>
#include <type_traits>
#include <variant>

namespace Fortran::parser::omp {

const OmpObjectList *GetOmpObjectList(const OmpClause &clause) {
  // Clauses with OmpObjectList as its data member
  using MemberObjectListClauses = std::tuple<OmpClause::Copyin,
      OmpClause::Copyprivate, OmpClause::Exclusive, OmpClause::Firstprivate,
      OmpClause::HasDeviceAddr, OmpClause::Inclusive, OmpClause::IsDevicePtr,
      OmpClause::Link, OmpClause::Private, OmpClause::Shared,
      OmpClause::UseDeviceAddr, OmpClause::UseDevicePtr>;

  // Clauses with OmpObjectList in the tuple
  using TupleObjectListClauses = std::tuple<OmpClause::AdjustArgs,
      OmpClause::Affinity, OmpClause::Aligned, OmpClause::Allocate,
      OmpClause::Enter, OmpClause::From, OmpClause::InReduction,
      OmpClause::Lastprivate, OmpClause::Linear, OmpClause::Map,
      OmpClause::Reduction, OmpClause::TaskReduction, OmpClause::To>;

  // TODO:: Generate the tuples using TableGen.
  return common::visit(
      common::visitors{
          [&](const OmpClause::Depend &x) -> const OmpObjectList * {
            if (auto *taskDep{std::get_if<OmpDependClause::TaskDep>(&x.v.u)}) {
              return &std::get<OmpObjectList>(taskDep->t);
            } else {
              return nullptr;
            }
          },
          [&](const auto &x) -> const OmpObjectList * {
            using Ty = std::decay_t<decltype(x)>;
            if constexpr (common::HasMember<Ty, MemberObjectListClauses>) {
              return &x.v;
            } else if constexpr (common::HasMember<Ty,
                                     TupleObjectListClauses>) {
              return &std::get<OmpObjectList>(x.v.t);
            } else {
              return nullptr;
            }
          },
      },
      clause.u);
}

const BlockConstruct *GetFortranBlockConstruct(
    const ExecutionPartConstruct &epc) {
  // ExecutionPartConstruct -> ExecutableConstruct
  //   -> Indirection<BlockConstruct>
  if (auto *ec{std::get_if<ExecutableConstruct>(&epc.u)}) {
    if (auto *ind{std::get_if<common::Indirection<BlockConstruct>>(&ec->u)}) {
      return &ind->value();
    }
  }
  return nullptr;
}

} // namespace Fortran::parser::omp
