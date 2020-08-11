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

#include "check-directive-structure.h"
#include "flang/Common/enum-set.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "llvm/Frontend/OpenACC/ACC.h.inc"

using AccDirectiveSet = Fortran::common::EnumSet<llvm::acc::Directive,
    llvm::acc::Directive_enumSize>;

using AccClauseSet =
    Fortran::common::EnumSet<llvm::acc::Clause, llvm::acc::Clause_enumSize>;

#define GEN_FLANG_DIRECTIVE_CLAUSE_SETS
#include "llvm/Frontend/OpenACC/ACC.cpp.inc"

namespace Fortran::semantics {

class AccStructureChecker
    : public DirectiveStructureChecker<llvm::acc::Directive, llvm::acc::Clause,
          parser::AccClause, llvm::acc::Clause_enumSize> {
public:
  AccStructureChecker(SemanticsContext &context)
      : DirectiveStructureChecker(context,
#define GEN_FLANG_DIRECTIVE_CLAUSE_MAP
#include "llvm/Frontend/OpenACC/ACC.cpp.inc"
        ) {
  }

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

  void CheckNoBranching(const parser::Block &block,
      const llvm::acc::Directive directive,
      const parser::CharBlock &directiveSource) const;

  llvm::StringRef getClauseName(llvm::acc::Clause clause) override;
  llvm::StringRef getDirectiveName(llvm::acc::Directive directive) override;
};

} // namespace Fortran::semantics

#endif // FORTRAN_SEMANTICS_CHECK_ACC_STRUCTURE_H_
