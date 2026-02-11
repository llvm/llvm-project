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
#include "flang/Parser/tools.h"

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

const OpenMPLoopConstruct *GetOmpLoop(const ExecutionPartConstruct &x) {
  if (auto *construct{GetOmp(x)}) {
    if (auto *omp{std::get_if<OpenMPLoopConstruct>(&construct->u)}) {
      return omp;
    }
  }
  return nullptr;
}
const DoConstruct *GetDoConstruct(const ExecutionPartConstruct &x) {
  if (auto *y{std::get_if<ExecutableConstruct>(&x.u)}) {
    if (auto *z{std::get_if<common::Indirection<DoConstruct>>(&y->u)}) {
      return &z->value();
    }
  }
  return nullptr;
}

// Get the Label from a Statement<...> contained in an ExecutionPartConstruct,
// or std::nullopt, if there is no Statement<...> contained in there.
template <typename T>
static std::optional<Label> GetStatementLabelHelper(const T &stmt) {
  if constexpr (IsStatement<T>::value) {
    return stmt.label;
  } else if constexpr (WrapperTrait<T>) {
    return GetStatementLabelHelper(stmt.v);
  } else if constexpr (UnionTrait<T>) {
    return common::visit(
        [&](auto &&s) { return GetStatementLabelHelper(s); }, stmt.u);
  }
  return std::nullopt;
}

std::optional<Label> GetStatementLabel(const ExecutionPartConstruct &x) {
  return GetStatementLabelHelper(x);
}

static std::optional<Label> GetFinalLabel(const Block &x) {
  if (!x.empty()) {
    const ExecutionPartConstruct &last{x.back()};
    if (auto *omp{Unwrap<OpenMPConstruct>(last)}) {
      return GetFinalLabel(*omp);
    } else if (auto *doLoop{Unwrap<DoConstruct>(last)}) {
      return GetFinalLabel(std::get<Block>(doLoop->t));
    } else {
      return GetStatementLabel(x.back());
    }
  } else {
    return std::nullopt;
  }
}

std::optional<Label> GetFinalLabel(const OpenMPConstruct &x) {
  return common::visit(
      [](auto &&s) -> std::optional<Label> {
        using TypeS = llvm::remove_cvref_t<decltype(s)>;
        if constexpr (std::is_same_v<TypeS, OpenMPSectionsConstruct>) {
          auto &list{std::get<std::list<OpenMPConstruct>>(s.t)};
          if (!list.empty()) {
            return GetFinalLabel(list.back());
          } else {
            return std::nullopt;
          }
        } else if constexpr ( //
            std::is_same_v<TypeS, OpenMPLoopConstruct> ||
            std::is_same_v<TypeS, OpenMPSectionConstruct> ||
            std::is_base_of_v<OmpBlockConstruct, TypeS>) {
          return GetFinalLabel(std::get<Block>(s.t));
        } else {
          return std::nullopt;
        }
      },
      x.u);
}

const OmpObjectList *GetOmpObjectList(const OmpClause &clause) {
  return common::visit([](auto &&s) { return GetOmpObjectList(s); }, clause.u);
}

const OmpObjectList *GetOmpObjectList(const OmpClause::Depend &clause) {
  return common::visit(
      common::visitors{
          [](const OmpDoacross &) -> const OmpObjectList * { return nullptr; },
          [](const OmpDependClause::TaskDep &x) { return GetOmpObjectList(x); },
      },
      clause.v.u);
}

const OmpObjectList *GetOmpObjectList(const OmpDependClause::TaskDep &x) {
  return &std::get<OmpObjectList>(x.t);
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

const OmpCombinerExpression *GetCombinerExpr(const OmpReductionSpecifier &x) {
  return addr_if(std::get<std::optional<OmpCombinerExpression>>(x.t));
}

const OmpCombinerExpression *GetCombinerExpr(const OmpClause &x) {
  if (auto *wrapped{std::get_if<OmpClause::Combiner>(&x.u)}) {
    return &wrapped->v.v;
  }
  return nullptr;
}

const OmpInitializerExpression *GetInitializerExpr(const OmpClause &x) {
  if (auto *wrapped{std::get_if<OmpClause::Initializer>(&x.u)}) {
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

void ExecutionPartIterator::step() {
  // Advance the iterator to the next legal position. If the current
  // position is a DO-loop or a loop construct, step into it.
  if (valid()) {
    IteratorType where{at()};
    if (auto *loop{GetOmpLoop(*where)}) {
      stack_.emplace_back(std::get<Block>(loop->t), &*where);
    } else if (auto *loop{GetDoConstruct(*where)}) {
      stack_.emplace_back(std::get<Block>(loop->t), &*where);
    } else {
      stack_.back().range =
          IteratorRange(std::next(where), stack_.back().range.end());
    }
    adjust();
  }
}

void ExecutionPartIterator::next() {
  // Advance the iterator to the next legal position. If the current
  // position is a DO-loop or a loop construct, step over it.
  if (valid()) {
    stack_.back().range =
        IteratorRange(std::next(at()), stack_.back().range.end());
    adjust();
  }
}

void ExecutionPartIterator::adjust() {
  // If the iterator is not at a legal location, keep advancing it until
  // it lands at a legal location or becomes invalid.
  while (valid()) {
    if (stack_.back().range.empty()) {
      stack_.pop_back();
      if (valid()) {
        stack_.back().range =
            IteratorRange(std::next(at()), stack_.back().range.end());
      }
    } else if (auto *block{GetFortranBlockConstruct(*at())}) {
      stack_.emplace_back(std::get<Block>(block->t), &*at());
    } else {
      break;
    }
  }
}

bool LoopNestIterator::isLoop(const ExecutionPartConstruct &c) {
  return Unwrap<OpenMPLoopConstruct>(c) != nullptr ||
      Unwrap<DoConstruct>(c) != nullptr;
}

} // namespace Fortran::parser::omp
