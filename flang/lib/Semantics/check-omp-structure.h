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
#include "llvm/Frontend/OpenMP/OMP.cpp.inc"

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
} // namespace omp
} // namespace llvm

namespace Fortran::semantics {

class OmpStructureChecker
    : public DirectiveStructureChecker<llvm::omp::Directive, llvm::omp::Clause,
          parser::OmpClause, llvm::omp::Clause_enumSize> {
public:
  OmpStructureChecker(SemanticsContext &context)
      : DirectiveStructureChecker(context,
#define GEN_FLANG_DIRECTIVE_CLAUSE_MAP
#include "llvm/Frontend/OpenMP/OMP.cpp.inc"
        ) {
  }

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
  void Enter(const parser::OpenMPDeclareTargetConstruct &);
  void Leave(const parser::OpenMPDeclareTargetConstruct &);

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

  void Leave(const parser::OmpClauseList &);
  void Enter(const parser::OmpClause &);
  void Enter(const parser::OmpNowait &);
  void Enter(const parser::OmpClause::Inbranch &);
  void Enter(const parser::OmpClause::Mergeable &);
  void Enter(const parser::OmpClause::Nogroup &);
  void Enter(const parser::OmpClause::Notinbranch &);
  void Enter(const parser::OmpClause::Untied &);
  void Enter(const parser::OmpClause::Collapse &);
  void Enter(const parser::OmpClause::Copyin &);
  void Enter(const parser::OmpClause::Copyprivate &);
  void Enter(const parser::OmpClause::Device &);
  void Enter(const parser::OmpClause::Final &);
  void Enter(const parser::OmpClause::Firstprivate &);
  void Enter(const parser::OmpClause::From &);
  void Enter(const parser::OmpClause::Grainsize &);
  void Enter(const parser::OmpClause::Lastprivate &);
  void Enter(const parser::OmpClause::NumTasks &);
  void Enter(const parser::OmpClause::NumTeams &);
  void Enter(const parser::OmpClause::NumThreads &);
  void Enter(const parser::OmpClause::Ordered &);
  void Enter(const parser::OmpClause::Priority &);
  void Enter(const parser::OmpClause::Private &);
  void Enter(const parser::OmpClause::Safelen &);
  void Enter(const parser::OmpClause::Shared &);
  void Enter(const parser::OmpClause::Simdlen &);
  void Enter(const parser::OmpClause::ThreadLimit &);
  void Enter(const parser::OmpClause::To &);
  void Enter(const parser::OmpClause::Link &);
  void Enter(const parser::OmpClause::Uniform &);
  void Enter(const parser::OmpClause::UseDevicePtr &);
  void Enter(const parser::OmpClause::IsDevicePtr &);
  // Memory-order-clause
  void Enter(const parser::OmpClause::SeqCst &);
  void Enter(const parser::OmpClause::AcqRel &);
  void Enter(const parser::OmpClause::Release &);
  void Enter(const parser::OmpClause::Acquire &);
  void Enter(const parser::OmpClause::Relaxed &);

  void Enter(const parser::OmpAlignedClause &);
  void Enter(const parser::OmpAllocateClause &);
  void Enter(const parser::OmpDefaultClause &);
  void Enter(const parser::OmpDefaultmapClause &);
  void Enter(const parser::OmpDependClause &);
  void Enter(const parser::OmpDistScheduleClause &);
  void Enter(const parser::OmpIfClause &);
  void Enter(const parser::OmpLinearClause &);
  void Enter(const parser::OmpMapClause &);
  void Enter(const parser::OmpProcBindClause &);
  void Enter(const parser::OmpReductionClause &);
  void Enter(const parser::OmpScheduleClause &);

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
  void GetSymbolsInObjectList(
      const parser::OmpObjectList &, std::vector<const Symbol *> &);
};
} // namespace Fortran::semantics
#endif // FORTRAN_SEMANTICS_CHECK_OMP_STRUCTURE_H_
