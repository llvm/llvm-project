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
    Directive::OMPD_distribute_parallel_do_simd, Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd, Directive::OMPD_do,
    Directive::OMPD_do_simd, Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd};
static OmpDirectiveSet doSimdSet{Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_parallel_do_simd, Directive::OMPD_do_simd,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do_simd};
static OmpDirectiveSet workShareSet{
    OmpDirectiveSet{Directive::OMPD_workshare,
        Directive::OMPD_parallel_workshare, Directive::OMPD_parallel_sections,
        Directive::OMPD_sections, Directive::OMPD_single} |
    doSet};
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
static OmpDirectiveSet teamSet{Directive::OMPD_teams,
    Directive::OMPD_teams_distribute,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_for,
    Directive::OMPD_teams_distribute_parallel_for_simd,
    Directive::OMPD_teams_distribute_simd};
static OmpDirectiveSet taskGeneratingSet{
    OmpDirectiveSet{Directive::OMPD_task} | taskloopSet};
static OmpDirectiveSet nestedOrderedErrSet{Directive::OMPD_critical,
    Directive::OMPD_ordered, Directive::OMPD_atomic, Directive::OMPD_task,
    Directive::OMPD_taskloop};
static OmpDirectiveSet nestedWorkshareErrSet{
    OmpDirectiveSet{Directive::OMPD_task, Directive::OMPD_taskloop,
        Directive::OMPD_critical, Directive::OMPD_ordered,
        Directive::OMPD_atomic, Directive::OMPD_master} |
    workShareSet};
static OmpDirectiveSet nestedMasterErrSet{
    OmpDirectiveSet{llvm::omp::Directive::OMPD_atomic} | taskGeneratingSet |
    workShareSet};
static OmpDirectiveSet nestedBarrierErrSet{
    OmpDirectiveSet{Directive::OMPD_critical, Directive::OMPD_ordered,
        Directive::OMPD_atomic, Directive::OMPD_master} |
    taskGeneratingSet | workShareSet};
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
  void Leave(const parser::OmpBeginBlockDirective &);
  void Enter(const parser::OmpEndBlockDirective &);
  void Leave(const parser::OmpEndBlockDirective &);

  void Enter(const parser::OpenMPSectionsConstruct &);
  void Leave(const parser::OpenMPSectionsConstruct &);
  void Enter(const parser::OmpEndSectionsDirective &);
  void Leave(const parser::OmpEndSectionsDirective &);

  void Enter(const parser::OpenMPDeclareSimdConstruct &);
  void Leave(const parser::OpenMPDeclareSimdConstruct &);
  void Enter(const parser::OpenMPDeclarativeAllocate &);
  void Leave(const parser::OpenMPDeclarativeAllocate &);
  void Enter(const parser::OpenMPDeclareTargetConstruct &);
  void Leave(const parser::OpenMPDeclareTargetConstruct &);
  void Enter(const parser::OpenMPExecutableAllocate &);
  void Leave(const parser::OpenMPExecutableAllocate &);
  void Enter(const parser::OpenMPThreadprivate &);
  void Leave(const parser::OpenMPThreadprivate &);

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

  // Get the OpenMP Clause Kind for the corresponding Parser class
  template <typename A>
  llvm::omp::Clause GetClauseKindForParserClass(const A &) {
#define GEN_FLANG_CLAUSE_PARSER_KIND_MAP
#include "llvm/Frontend/OpenMP/OMP.inc"
  }

private:
  bool HasInvalidWorksharingNesting(
      const parser::CharBlock &, const OmpDirectiveSet &);
  bool IsCloselyNestedRegion(const OmpDirectiveSet &set);
  void HasInvalidTeamsNesting(
      const llvm::omp::Directive &dir, const parser::CharBlock &source);
  void HasInvalidDistributeNesting(const parser::OpenMPLoopConstruct &x);
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
  bool IsDataRefTypeParamInquiry(const parser::DataRef *dataRef);
  void CheckIsVarPartOfAnotherVar(
      const parser::CharBlock &source, const parser::OmpObjectList &objList);
  void CheckThreadprivateOrDeclareTargetVar(
      const parser::OmpObjectList &objList);
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
  bool CheckTargetBlockOnlyTeams(const parser::Block &);
  void CheckWorkshareBlockStmts(const parser::Block &, parser::CharBlock);

  void CheckLoopItrVariableIsInt(const parser::OpenMPLoopConstruct &x);
  void CheckDoWhile(const parser::OpenMPLoopConstruct &x);
  void CheckCycleConstraints(const parser::OpenMPLoopConstruct &x);
  template <typename T, typename D> bool IsOperatorValid(const T &, const D &);
  void CheckAtomicMemoryOrderClause(
      const parser::OmpAtomicClauseList &, const parser::OmpAtomicClauseList &);
  void CheckAtomicMemoryOrderClause(const parser::OmpAtomicClauseList &);
  void CheckAtomicUpdateAssignmentStmt(const parser::AssignmentStmt &);
  void CheckAtomicConstructStructure(const parser::OpenMPAtomicConstruct &);
  void CheckDistLinear(const parser::OpenMPLoopConstruct &x);
  void CheckSIMDNest(const parser::OpenMPConstruct &x);
  void CheckTargetNest(const parser::OpenMPConstruct &x);
  void CheckCancellationNest(
      const parser::CharBlock &source, const parser::OmpCancelType::Type &type);
  std::int64_t GetOrdCollapseLevel(const parser::OpenMPLoopConstruct &x);
  void CheckIfDoOrderedClause(const parser::OmpBlockDirective &blkDirectiv);
  bool CheckReductionOperators(const parser::OmpClause::Reduction &);
  bool CheckIntrinsicOperator(
      const parser::DefinedOperator::IntrinsicOperator &);
  void CheckReductionTypeList(const parser::OmpClause::Reduction &);
  void CheckMasterNesting(const parser::OpenMPBlockConstruct &x);
  void ChecksOnOrderedAsBlock();
  void CheckBarrierNesting(const parser::OpenMPSimpleStandaloneConstruct &x);
  void ChecksOnOrderedAsStandalone();
  void CheckReductionArraySection(const parser::OmpObjectList &ompObjectList);
  void CheckIntentInPointerAndDefinable(
      const parser::OmpObjectList &, const llvm::omp::Clause);
  void CheckArraySection(const parser::ArrayElement &arrayElement,
      const parser::Name &name, const llvm::omp::Clause clause);
  void CheckMultipleAppearanceAcrossContext(
      const parser::OmpObjectList &ompObjectList);
  const parser::OmpObjectList *GetOmpObjectList(const parser::OmpClause &);
  void CheckPredefinedAllocatorRestriction(const parser::CharBlock &source,
      const parser::OmpObjectList &ompObjectList);
  void CheckPredefinedAllocatorRestriction(
      const parser::CharBlock &source, const parser::Name &name);
  bool isPredefinedAllocator{false};
  void EnterDirectiveNest(const int index) { directiveNest_[index]++; }
  void ExitDirectiveNest(const int index) { directiveNest_[index]--; }
  int GetDirectiveNest(const int index) { return directiveNest_[index]; }

  enum directiveNestType {
    SIMDNest,
    TargetBlockOnlyTeams,
    TargetNest,
    LastType
  };
  int directiveNest_[LastType + 1] = {0};
};
} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_
