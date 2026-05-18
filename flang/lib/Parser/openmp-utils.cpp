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
#include "llvm/ADT/StringRef.h"
#include "llvm/Frontend/OpenMP/OMP.h"

#include <tuple>
#include <type_traits>
#include <variant>

namespace Fortran::parser::omp {

const parser::Designator *GetDesignatorFromObj(
    const parser::OmpObject &object) {
  return std::get_if<parser::Designator>(&object.u);
}

const parser::DataRef *GetDataRefFromObj(const parser::OmpObject &object) {
  if (auto *desg{GetDesignatorFromObj(object)}) {
    return std::get_if<parser::DataRef>(&desg->u);
  }
  return nullptr;
}

const parser::ArrayElement *GetArrayElementFromObj(
    const parser::OmpObject &object) {
  if (auto *dataRef{GetDataRefFromObj(object)}) {
    using ElementIndirection = common::Indirection<parser::ArrayElement>;
    if (auto *ind{std::get_if<ElementIndirection>(&dataRef->u)}) {
      return &ind->value();
    }
  }
  return nullptr;
}

std::optional<parser::CharBlock> GetObjectSource(
    const parser::OmpObject &object) {
  if (auto *name{std::get_if<parser::Name>(&object.u)}) {
    return name->source;
  } else if (auto *desg{std::get_if<parser::Designator>(&object.u)}) {
    return GetLastName(*desg).source;
  }
  return std::nullopt;
}

const parser::OmpObject *GetArgumentObject(
    const parser::OmpArgument &argument) {
  if (auto *locator{std::get_if<parser::OmpLocator>(&argument.u)}) {
    return std::get_if<parser::OmpObject>(&locator->u);
  }
  return nullptr;
}

namespace detail {
struct DirectiveSpecificationScope {
  using ODS = OmpDirectiveSpecification;
  template <typename T> static const ODS &GetODS(const T &x) {
    if constexpr ( //
        std::is_base_of_v<OmpBlockConstruct, T> ||
        std::is_same_v<OpenMPSectionsConstruct, T>) {
      return x.BeginDir();
    } else if constexpr (WrapperTrait<T>) {
      return GetODS(x.v);
    } else if constexpr (UnionTrait<T>) {
      return std::visit(
          [](auto &&s) -> decltype(auto) { return GetODS(s); }, x.u);
    } else {
      static_assert(std::is_same_v<OpenMPSectionConstruct, T>);
      llvm_unreachable("This function does not work for SECTION");
    }
  }
  static inline const ODS &GetODS(const ODS &x) { return x; }
};
} // namespace detail

const OmpDirectiveSpecification &GetOmpDirectiveSpecification(
    const OpenMPConstruct &x) {
  return std::visit(
      [](auto &&s) -> decltype(auto) {
        return detail::DirectiveSpecificationScope::GetODS(s);
      },
      x.u);
}

const OmpDirectiveSpecification &GetOmpDirectiveSpecification(
    const OpenMPDeclarativeConstruct &x) {
  return std::visit(
      [](auto &&s) -> decltype(auto) {
        return detail::DirectiveSpecificationScope::GetODS(s);
      },
      x.u);
}

std::string GetUpperName(llvm::omp::Clause id, unsigned version) {
  llvm::StringRef name{llvm::omp::getOpenMPClauseName(id, version)};
  return parser::ToUpperCaseLetters(name);
}

std::string GetUpperName(llvm::omp::Directive id, unsigned version) {
  llvm::StringRef name{llvm::omp::getOpenMPDirectiveName(id, version)};
  return parser::ToUpperCaseLetters(name);
}

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

const OmpClause *FindClause(
    const OmpDirectiveSpecification &spec, llvm::omp::Clause clauseId) {
  for (auto &clause : spec.Clauses().v) {
    if (clause.Id() == clauseId) {
      return &clause;
    }
  }
  return nullptr;
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
      ++stack_.back().location.at;
    }
    adjust();
  }
}

void ExecutionPartIterator::next() {
  // Advance the iterator to the next legal position. If the current
  // position is a DO-loop or a loop construct, step over it.
  if (valid()) {
    ++stack_.back().location.at;
    adjust();
  }
}

void ExecutionPartIterator::adjust() {
  // If the iterator is not at a legal location, keep advancing it until
  // it lands at a legal location or becomes invalid.
  while (valid()) {
    if (stack_.back().location.atEnd()) {
      stack_.pop_back();
      if (valid()) {
        ++stack_.back().location.at;
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
