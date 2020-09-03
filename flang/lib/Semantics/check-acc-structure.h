//===-- lib/Semantics/check-acc-structure.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// OpenACC structure validity check list
//    1. invalid clauses on directive
//    2. invalid repeated clauses on directive
//    3. invalid nesting of regions
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_CHECK_ACC_STRUCTURE_H_
#define FORTRAN_SEMANTICS_CHECK_ACC_STRUCTURE_H_

#include "flang/Common/enum-set.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "llvm/Frontend/OpenACC/ACC.h.inc"

#include <unordered_map>

using AccDirectiveSet = Fortran::common::EnumSet<llvm::acc::Directive,
    llvm::acc::Directive_enumSize>;

using AccClauseSet =
    Fortran::common::EnumSet<llvm::acc::Clause, llvm::acc::Clause_enumSize>;

#define GEN_FLANG_DIRECTIVE_CLAUSE_SETS
#include "llvm/Frontend/OpenACC/ACC.cpp.inc"

namespace Fortran::semantics {

class AccStructureChecker : public virtual BaseChecker {
public:
  AccStructureChecker(SemanticsContext &context) : context_{context} {}

  // Construct and directives
  void Enter(const parser::OpenACCBlockConstruct &);
  void Leave(const parser::OpenACCBlockConstruct &);
  void Enter(const parser::OpenACCCombinedConstruct &);
  void Leave(const parser::OpenACCCombinedConstruct &);
  void Enter(const parser::OpenACCLoopConstruct &);
  void Leave(const parser::OpenACCLoopConstruct &);
  void Enter(const parser::OpenACCRoutineConstruct &);
  void Leave(const parser::OpenACCRoutineConstruct &);
  void Enter(const parser::OpenACCStandaloneConstruct &);
  void Leave(const parser::OpenACCStandaloneConstruct &);
  void Enter(const parser::OpenACCStandaloneDeclarativeConstruct &);
  void Leave(const parser::OpenACCStandaloneDeclarativeConstruct &);

  // Clauses
  void Leave(const parser::AccClauseList &);
  void Enter(const parser::AccClause &);

  void Enter(const parser::AccClause::Auto &);
  void Enter(const parser::AccClause::Async &);
  void Enter(const parser::AccClause::Attach &);
  void Enter(const parser::AccClause::Bind &);
  void Enter(const parser::AccClause::Capture &);
  void Enter(const parser::AccClause::Create &);
  void Enter(const parser::AccClause::Collapse &);
  void Enter(const parser::AccClause::Copy &);
  void Enter(const parser::AccClause::Copyin &);
  void Enter(const parser::AccClause::Copyout &);
  void Enter(const parser::AccClause::Default &);
  void Enter(const parser::AccClause::DefaultAsync &);
  void Enter(const parser::AccClause::Delete &);
  void Enter(const parser::AccClause::Detach &);
  void Enter(const parser::AccClause::Device &);
  void Enter(const parser::AccClause::DeviceNum &);
  void Enter(const parser::AccClause::DevicePtr &);
  void Enter(const parser::AccClause::DeviceResident &);
  void Enter(const parser::AccClause::DeviceType &);
  void Enter(const parser::AccClause::Finalize &);
  void Enter(const parser::AccClause::FirstPrivate &);
  void Enter(const parser::AccClause::Gang &);
  void Enter(const parser::AccClause::Host &);
  void Enter(const parser::AccClause::If &);
  void Enter(const parser::AccClause::IfPresent &);
  void Enter(const parser::AccClause::Independent &);
  void Enter(const parser::AccClause::Link &);
  void Enter(const parser::AccClause::NoCreate &);
  void Enter(const parser::AccClause::NoHost &);
  void Enter(const parser::AccClause::NumGangs &);
  void Enter(const parser::AccClause::NumWorkers &);
  void Enter(const parser::AccClause::Present &);
  void Enter(const parser::AccClause::Private &);
  void Enter(const parser::AccClause::Read &);
  void Enter(const parser::AccClause::Reduction &);
  void Enter(const parser::AccClause::Self &);
  void Enter(const parser::AccClause::Seq &);
  void Enter(const parser::AccClause::Tile &);
  void Enter(const parser::AccClause::UseDevice &);
  void Enter(const parser::AccClause::Vector &);
  void Enter(const parser::AccClause::VectorLength &);
  void Enter(const parser::AccClause::Wait &);
  void Enter(const parser::AccClause::Worker &);
  void Enter(const parser::AccClause::Write &);

private:
#define GEN_FLANG_DIRECTIVE_CLAUSE_MAP
#include "llvm/Frontend/OpenACC/ACC.cpp.inc"

  struct AccContext {
    AccContext(parser::CharBlock source, llvm::acc::Directive d)
        : directiveSource{source}, directive{d} {}

    parser::CharBlock directiveSource{nullptr};
    parser::CharBlock clauseSource{nullptr};
    llvm::acc::Directive directive;
    AccClauseSet allowedClauses{};
    AccClauseSet allowedOnceClauses{};
    AccClauseSet allowedExclusiveClauses{};
    AccClauseSet requiredClauses{};

    const parser::AccClause *clause{nullptr};
    std::multimap<llvm::acc::Clause, const parser::AccClause *> clauseInfo;
    std::list<llvm::acc::Clause> actualClauses;
  };

  // back() is the top of the stack
  AccContext &GetContext() {
    CHECK(!accContext_.empty());
    return accContext_.back();
  }

  void SetContextClause(const parser::AccClause &clause) {
    GetContext().clauseSource = clause.source;
    GetContext().clause = &clause;
  }

  void SetContextClauseInfo(llvm::acc::Clause type) {
    GetContext().clauseInfo.emplace(type, GetContext().clause);
  }

  void AddClauseToCrtContext(llvm::acc::Clause type) {
    GetContext().actualClauses.push_back(type);
  }

  const parser::AccClause *FindClause(llvm::acc::Clause type) {
    auto it{GetContext().clauseInfo.find(type)};
    if (it != GetContext().clauseInfo.end()) {
      return it->second;
    }
    return nullptr;
  }

  void PushContext(const parser::CharBlock &source, llvm::acc::Directive dir) {
    accContext_.emplace_back(source, dir);
  }

  void SetClauseSets(llvm::acc::Directive dir) {
    accContext_.back().allowedClauses = directiveClausesTable[dir].allowed;
    accContext_.back().allowedOnceClauses =
        directiveClausesTable[dir].allowedOnce;
    accContext_.back().allowedExclusiveClauses =
        directiveClausesTable[dir].allowedExclusive;
    accContext_.back().requiredClauses =
        directiveClausesTable[dir].requiredOneOf;
  }
  void PushContextAndClauseSets(
      const parser::CharBlock &source, llvm::acc::Directive dir) {
    PushContext(source, dir);
    SetClauseSets(dir);
  }

  void SayNotMatching(const parser::CharBlock &, const parser::CharBlock &);

  template <typename B> void CheckMatching(const B &beginDir, const B &endDir) {
    const auto &begin{beginDir.v};
    const auto &end{endDir.v};
    if (begin != end) {
      SayNotMatching(beginDir.source, endDir.source);
    }
  }

  // Check that only clauses in set are after the specific clauses.
  void CheckOnlyAllowedAfter(llvm::acc::Clause clause, AccClauseSet set);
  void CheckRequireAtLeastOneOf();
  void CheckAllowed(llvm::acc::Clause clause);
  void CheckAtLeastOneClause();
  void CheckNotAllowedIfClause(llvm::acc::Clause clause, AccClauseSet set);
  std::string ContextDirectiveAsFortran();

  void CheckNoBranching(const parser::Block &block,
      const llvm::acc::Directive directive,
      const parser::CharBlock &directiveSource) const;

  void RequiresConstantPositiveParameter(
      const llvm::acc::Clause &clause, const parser::ScalarIntConstantExpr &i);
  void OptionalConstantPositiveParameter(const llvm::acc::Clause &clause,
      const std::optional<parser::ScalarIntConstantExpr> &o);

  SemanticsContext &context_;
  std::vector<AccContext> accContext_; // used as a stack

  std::string ClauseSetToString(const AccClauseSet set);
};

} // namespace Fortran::semantics

#endif // FORTRAN_SEMANTICS_CHECK_ACC_STRUCTURE_H_
