//===-- lib/Semantics/rewrite-directives.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "rewrite-directives.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include "llvm/Frontend/OpenMP/OMP.h"
#include <list>

namespace Fortran::semantics {

using namespace parser::literals;

class DirectiveRewriteMutator {
public:
  explicit DirectiveRewriteMutator(SemanticsContext &context)
      : context_{context} {}

  // Default action for a parse tree node is to visit children.
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}

protected:
  SemanticsContext &context_;
};

// Rewrite atomic constructs to add an explicit memory ordering to all that do
// not specify it, honoring in this way the `atomic_default_mem_order` clause of
// the REQUIRES directive.
class OmpRewriteMutator : public DirectiveRewriteMutator {
public:
  explicit OmpRewriteMutator(SemanticsContext &context)
      : DirectiveRewriteMutator(context) {}

  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}

  bool Pre(parser::OpenMPAtomicConstruct &);
  bool Pre(parser::OpenMPRequiresConstruct &);

private:
  bool atomicDirectiveDefaultOrderFound_{false};
};

bool OmpRewriteMutator::Pre(parser::OpenMPAtomicConstruct &x) {
  // Find top-level parent of the operation.
  Symbol *topLevelParent{[&]() {
    Symbol *symbol{nullptr};
    Scope *scope{&context_.FindScope(
        std::get<parser::OmpDirectiveSpecification>(x.t).source)};
    do {
      if (Symbol * parent{scope->symbol()}) {
        symbol = parent;
      }
      scope = &scope->parent();
    } while (!scope->IsGlobal());

    assert(symbol &&
        "Atomic construct must be within a scope associated with a symbol");
    return symbol;
  }()};

  // Get the `atomic_default_mem_order` clause from the top-level parent.
  std::optional<common::OmpMemoryOrderType> defaultMemOrder;
  common::visit(
      [&](auto &details) {
        if constexpr (std::is_convertible_v<decltype(&details),
                          WithOmpDeclarative *>) {
          if (details.has_ompAtomicDefaultMemOrder()) {
            defaultMemOrder = *details.ompAtomicDefaultMemOrder();
          }
        }
      },
      topLevelParent->details());

  if (!defaultMemOrder) {
    return false;
  }

  auto findMemOrderClause{[](const parser::OmpClauseList &clauses) {
    return llvm::any_of(
        clauses.v, [](auto &clause) -> const parser::OmpClause * {
          switch (clause.Id()) {
          case llvm::omp::Clause::OMPC_acq_rel:
          case llvm::omp::Clause::OMPC_acquire:
          case llvm::omp::Clause::OMPC_relaxed:
          case llvm::omp::Clause::OMPC_release:
          case llvm::omp::Clause::OMPC_seq_cst:
            return &clause;
          default:
            return nullptr;
          }
        });
  }};

  auto &dirSpec{std::get<parser::OmpDirectiveSpecification>(x.t)};
  auto &clauseList{std::get<std::optional<parser::OmpClauseList>>(dirSpec.t)};
  if (clauseList) {
    if (findMemOrderClause(*clauseList)) {
      return false;
    }
  } else {
    clauseList = parser::OmpClauseList(decltype(parser::OmpClauseList::v){});
  }

  // Add a memory order clause to the atomic directive.
  atomicDirectiveDefaultOrderFound_ = true;
  llvm::omp::Clause kind{x.GetKind()};
  switch (*defaultMemOrder) {
  case common::OmpMemoryOrderType::Acq_Rel:
    // FIXME: Implement 5.0 rules, pending clarification on later spec
    // versions.
    // [5.0:62:22-26]
    if (kind == llvm::omp::Clause::OMPC_read) {
      clauseList->v.emplace_back(
          parser::OmpClause{parser::OmpClause::Acquire{}});
    } else if (kind == llvm::omp::Clause::OMPC_update && x.IsCapture()) {
      clauseList->v.emplace_back(
          parser::OmpClause{parser::OmpClause::AcqRel{}});
    } else {
      clauseList->v.emplace_back(
          parser::OmpClause{parser::OmpClause::Release{}});
    }
    break;
  case common::OmpMemoryOrderType::Relaxed:
    clauseList->v.emplace_back(parser::OmpClause{parser::OmpClause::Relaxed{}});
    break;
  case common::OmpMemoryOrderType::Seq_Cst:
    clauseList->v.emplace_back(parser::OmpClause{parser::OmpClause::SeqCst{}});
    break;
  default:
    // FIXME: Don't process other values at the moment since their validity
    // depends on the OpenMP version (which is unavailable here).
    break;
  }

  return false;
}

bool OmpRewriteMutator::Pre(parser::OpenMPRequiresConstruct &x) {
  for (parser::OmpClause &clause : std::get<parser::OmpClauseList>(x.t).v) {
    if (std::holds_alternative<parser::OmpClause::AtomicDefaultMemOrder>(
            clause.u) &&
        atomicDirectiveDefaultOrderFound_) {
      context_.Say(clause.source,
          "REQUIRES directive with '%s' clause found lexically after atomic "
          "operation without a memory order clause"_err_en_US,
          parser::ToUpperCaseLetters(llvm::omp::getOpenMPClauseName(
              llvm::omp::OMPC_atomic_default_mem_order)
                                         .str()));
    }
  }
  return false;
}

bool RewriteOmpParts(SemanticsContext &context, parser::Program &program) {
  if (!context.IsEnabled(common::LanguageFeature::OpenMP)) {
    return true;
  }
  OmpRewriteMutator ompMutator{context};
  parser::Walk(program, ompMutator);
  return !context.AnyFatalError();
}

} // namespace Fortran::semantics
