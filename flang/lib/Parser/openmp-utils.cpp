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

const OpenMPDeclarativeConstruct *GetOmp(const DeclarationConstruct &x) {
  if (auto *y = std::get_if<SpecificationConstruct>(&x.u)) {
    if (auto *z{std::get_if<common::Indirection<OpenMPDeclarativeConstruct>>(
            &y->u)}) {
      return &z->value();
    }
  }
  return nullptr;
}

const OpenMPConstruct *GetOmp(const ExecutionPartConstruct &x) {
  if (auto *y{std::get_if<ExecutableConstruct>(&x.u)}) {
    if (auto *z{std::get_if<common::Indirection<OpenMPConstruct>>(&y->u)}) {
      return &z->value();
    }
  }
  return nullptr;
}

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

/// parser::Block is a list of executable constructs, parser::BlockConstruct
/// is Fortran's BLOCK/ENDBLOCK construct.
/// Strip the outermost BlockConstructs, return the reference to the Block
/// in the executable part of the innermost of the stripped constructs.
/// Specifically, if the given `block` has a single entry (it's a list), and
/// the entry is a BlockConstruct, get the Block contained within. Repeat
/// this step as many times as possible.
const Block &GetInnermostExecPart(const Block &block) {
  const Block *iter{&block};
  while (iter->size() == 1) {
    const ExecutionPartConstruct &ep{iter->front()};
    if (auto *bc{GetFortranBlockConstruct(ep)}) {
      iter = &std::get<Block>(bc->t);
    } else {
      break;
    }
  }
  return *iter;
}

bool IsStrictlyStructuredBlock(const Block &block) {
  if (block.size() == 1) {
    return GetFortranBlockConstruct(block.front()) != nullptr;
  } else {
    return false;
  }
}

const OmpCombinerExpression *GetCombinerExpr(
    const OmpReductionSpecifier &rspec) {
  return addr_if(std::get<std::optional<OmpCombinerExpression>>(rspec.t));
}

const OmpInitializerExpression *GetInitializerExpr(const OmpClause &init) {
  if (auto *wrapped{std::get_if<OmpClause::Initializer>(&init.u)}) {
    return &wrapped->v.v;
  }
  return nullptr;
}

static void SplitOmpAllocateHelper(
    OmpAllocateInfo &n, const OmpAllocateDirective &x) {
  n.dirs.push_back(&x);
  const Block &body{std::get<Block>(x.t)};
  if (!body.empty()) {
    if (auto *omp{GetOmp(body.front())}) {
      if (auto *ad{std::get_if<OmpAllocateDirective>(&omp->u)}) {
        return SplitOmpAllocateHelper(n, *ad);
      }
    }
    n.body = &body.front();
  }
}

OmpAllocateInfo SplitOmpAllocate(const OmpAllocateDirective &x) {
  OmpAllocateInfo info;
  SplitOmpAllocateHelper(info, x);
  return info;
}

} // namespace Fortran::parser::omp
