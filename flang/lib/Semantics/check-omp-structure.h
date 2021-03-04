//===-- lib/Semantics/check-omp-structure.h ---------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// OpenMP structure validity check list
//    1. invalid clauses on directive
//    2. invalid repeated clauses on directive
//    3. TODO: invalid nesting of regions

#ifndef FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_
#define FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_

#include "check-directive-structure.h"
#include "flang/Common/enum-set.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using OmpDirectiveSet = Fortran::common::EnumSet<llvm::omp::Directive,
    llvm::omp::Directive_enumSize>;

using OmpClauseSet =
    Fortran::common::EnumSet<llvm::omp::Clause, llvm::omp::Clause_enumSize>;

#define GEN_FLANG_DIRECTIVE_CLAUSE_SETS
#include "llvm/Frontend/OpenMP/OMP.inc"

namespace llvm {
namespace omp {
static OmpDirectiveSet parallelSet{Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd, Directive::OMPD_parallel,
    Directive::OMPD_parallel_do, Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_sections, Directive::OMPD_parallel_workshare,
    Directive::OMPD_target_parallel, Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd};
static OmpDirectiveSet doSet{Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd, Directive::OMPD_parallel,
    Directive::OMPD_parallel_do, Directive::OMPD_parallel_do_simd,
    Directive::OMPD_do, Directive::OMPD_do_simd,
    Directive::OMPD_target_parallel_do, Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd};
static OmpDirectiveSet doSimdSet{Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_parallel_do_simd, Directive::OMPD_do_simd,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do_simd};
static OmpDirectiveSet taskloopSet{
    Directive::OMPD_taskloop, Directive::OMPD_taskloop_simd};
static OmpDirectiveSet targetSet{Directive::OMPD_target,
    Directive::OMPD_target_parallel, Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel_do_simd, Directive::OMPD_target_simd,
    Directive::OMPD_target_teams, Directive::OMPD_target_teams_distribute,
    Directive::OMPD_target_teams_distribute_simd};
static OmpDirectiveSet simdSet{Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd, Directive::OMPD_parallel_do_simd,
    Directive::OMPD_do_simd, Directive::OMPD_simd,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_simd, Directive::OMPD_target_simd,
    Directive::OMPD_taskloop_simd,
    Directive::OMPD_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_simd};
static OmpDirectiveSet taskGeneratingSet{
    OmpDirectiveSet{Directive::OMPD_task} | taskloopSet};
static OmpClauseSet privateSet{
    Clause::OMPC_private, Clause::OMPC_firstprivate, Clause::OMPC_lastprivate};
static OmpClauseSet privateReductionSet{
    OmpClauseSet{Clause::OMPC_reduction} | privateSet};
} // namespace omp
} // namespace llvm

namespace Fortran::semantics {

// Mapping from 'Symbol' to 'Source' to keep track of the variables
// used in multiple clauses
using SymbolSourceMap = std::multimap<const Symbol *, parser::CharBlock>;
// Multimap to check the triple <current_dir, enclosing_dir, enclosing_clause>
using DirectivesClauseTriple = std::multimap<llvm::omp::Directive,
    std::pair<llvm::omp::Directive, const OmpClauseSet>>;

class OmpStructureChecker
    : public DirectiveStructureChecker<llvm::omp::Directive, llvm::omp::Clause,
          parser::OmpClause, llvm::omp::Clause_enumSize> {
public:
  OmpStructureChecker(SemanticsContext &context)
      : DirectiveStructureChecker(context,
#define GEN_FLANG_DIRECTIVE_CLAUSE_MAP
#include "llvm/Frontend/OpenMP/OMP.inc"
        ) {
  }
  using llvmOmpClause = const llvm::omp::Clause;

  void Enter(const parser::OpenMPConstruct &);
  void Enter(const parser::OpenMPLoopConstruct &);
  void Leave(const parser::OpenMPLoopConstruct &);
  void Enter(const parser::OmpEndLoopDirective &);

  void Enter(const parser::OpenMPBlockConstruct &);
  void Leave(const parser::OpenMPBlockConstruct &);
  void Enter(const parser::OmpEndBlockDirective &);

  void Enter(const parser::OpenMPSectionsConstruct &);
  void Leave(const parser::OpenMPSectionsConstruct &);
  void Enter(const parser::OmpEndSectionsDirective &);

  void Enter(const parser::OpenMPDeclareSimdConstruct &);
  void Leave(const parser::OpenMPDeclareSimdConstruct &);
  void Enter(const parser::OpenMPDeclarativeAllocate &);
  void Leave(const parser::OpenMPDeclarativeAllocate &);
  void Enter(const parser::OpenMPDeclareTargetConstruct &);
  void Leave(const parser::OpenMPDeclareTargetConstruct &);
  void Enter(const parser::OpenMPExecutableAllocate &);
  void Leave(const parser::OpenMPExecutableAllocate &);

  void Enter(const parser::OpenMPSimpleStandaloneConstruct &);
  void Leave(const parser::OpenMPSimpleStandaloneConstruct &);
  void Enter(const parser::OpenMPFlushConstruct &);
  void Leave(const parser::OpenMPFlushConstruct &);
  void Enter(const parser::OpenMPCancelConstruct &);
  void Leave(const parser::OpenMPCancelConstruct &);
  void Enter(const parser::OpenMPCancellationPointConstruct &);
  void Leave(const parser::OpenMPCancellationPointConstruct &);
  void Enter(const parser::OpenMPCriticalConstruct &);
  void Leave(const parser::OpenMPCriticalConstruct &);
  void Enter(const parser::OpenMPAtomicConstruct &);
  void Leave(const parser::OpenMPAtomicConstruct &);

  void Leave(const parser::OmpClauseList &);
  void Enter(const parser::OmpClause &);

  void Enter(const parser::OmpAtomicRead &);
  void Leave(const parser::OmpAtomicRead &);
  void Enter(const parser::OmpAtomicWrite &);
  void Leave(const parser::OmpAtomicWrite &);
  void Enter(const parser::OmpAtomicUpdate &);
  void Leave(const parser::OmpAtomicUpdate &);
  void Enter(const parser::OmpAtomicCapture &);
  void Leave(const parser::OmpAtomic &);

#define GEN_FLANG_CLAUSE_CHECK_ENTER
#include "llvm/Frontend/OpenMP/OMP.inc"

private:
  bool HasInvalidWorksharingNesting(
      const parser::CharBlock &, const OmpDirectiveSet &);

  // specific clause related
  bool ScheduleModifierHasType(const parser::OmpScheduleClause &,
      const parser::OmpScheduleModifierType::ModType &);
  void CheckAllowedMapTypes(const parser::OmpMapType::Type &,
      const std::list<parser::OmpMapType::Type> &);
  llvm::StringRef getClauseName(llvm::omp::Clause clause) override;
  llvm::StringRef getDirectiveName(llvm::omp::Directive directive) override;

  void CheckDependList(const parser::DataRef &);
  void CheckDependArraySection(
      const common::Indirection<parser::ArrayElement> &, const parser::Name &);
  void CheckIsVarPartOfAnotherVar(const parser::OmpObjectList &objList);
  void CheckIntentInPointer(
      const parser::OmpObjectList &, const llvm::omp::Clause);
  void GetSymbolsInObjectList(const parser::OmpObjectList &, SymbolSourceMap &);
  void CheckDefinableObjects(SymbolSourceMap &, const llvm::omp::Clause);
  void CheckPrivateSymbolsInOuterCxt(
      SymbolSourceMap &, DirectivesClauseTriple &, const llvm::omp::Clause);
  const parser::Name GetLoopIndex(const parser::DoConstruct *x);
  void SetLoopInfo(const parser::OpenMPLoopConstruct &x);
  void CheckIsLoopIvPartOfClause(
      llvmOmpClause clause, const parser::OmpObjectList &ompObjectList);
  void CheckWorkshareBlockStmts(const parser::Block &, parser::CharBlock);

  void CheckLoopItrVariableIsInt(const parser::OpenMPLoopConstruct &x);
  void CheckDoWhile(const parser::OpenMPLoopConstruct &x);
  void CheckCycleConstraints(const parser::OpenMPLoopConstruct &x);
  std::int64_t GetOrdCollapseLevel(const parser::OpenMPLoopConstruct &x);
  void CheckIfDoOrderedClause(const parser::OmpBlockDirective &blkDirectiv);
};
} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_
