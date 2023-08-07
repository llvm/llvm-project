//===-- include/flang/Semantics/openmp-directive-sets.h ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_OPENMP_DIRECTIVE_SETS_H_
#define FORTRAN_SEMANTICS_OPENMP_DIRECTIVE_SETS_H_

#include "flang/Common/enum-set.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using OmpDirectiveSet = Fortran::common::EnumSet<llvm::omp::Directive,
    llvm::omp::Directive_enumSize>;

namespace llvm::omp {
//===----------------------------------------------------------------------===//
// Directive sets for single directives
//===----------------------------------------------------------------------===//
// - top<Directive>Set: The directive appears alone or as the first in a
//   combined construct.
// - all<Directive>Set: All standalone or combined uses of the directive.

const inline OmpDirectiveSet topParallelSet{
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
};

const inline OmpDirectiveSet allParallelSet{
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
    Directive::OMPD_target_parallel,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd,
};

const inline OmpDirectiveSet topDoSet{
    Directive::OMPD_do,
    Directive::OMPD_do_simd,
};

const inline OmpDirectiveSet allDoSet{
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_do,
    Directive::OMPD_do_simd,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd,
};

const inline OmpDirectiveSet topTaskloopSet{
    Directive::OMPD_taskloop,
    Directive::OMPD_taskloop_simd,
};

const inline OmpDirectiveSet allTaskloopSet = topTaskloopSet;

const inline OmpDirectiveSet topTargetSet{
    Directive::OMPD_target,
    Directive::OMPD_target_parallel,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_simd,
    Directive::OMPD_target_teams,
    Directive::OMPD_target_teams_distribute,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_simd,
};

const inline OmpDirectiveSet allTargetSet = topTargetSet;

const inline OmpDirectiveSet topSimdSet{
    Directive::OMPD_simd,
};

const inline OmpDirectiveSet allSimdSet{
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
    Directive::OMPD_do_simd,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_simd,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_simd,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_simd,
    Directive::OMPD_taskloop_simd,
    Directive::OMPD_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_simd,
};

const inline OmpDirectiveSet topTeamsSet{
    Directive::OMPD_teams,
    Directive::OMPD_teams_distribute,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_simd,
};

const inline OmpDirectiveSet allTeamsSet{
    llvm::omp::OMPD_target_teams,
    llvm::omp::OMPD_target_teams_distribute,
    llvm::omp::OMPD_target_teams_distribute_parallel_do,
    llvm::omp::OMPD_target_teams_distribute_parallel_do_simd,
    llvm::omp::OMPD_target_teams_distribute_simd,
    llvm::omp::OMPD_teams,
    llvm::omp::OMPD_teams_distribute,
    llvm::omp::OMPD_teams_distribute_parallel_do,
    llvm::omp::OMPD_teams_distribute_parallel_do_simd,
    llvm::omp::OMPD_teams_distribute_simd,
};

const inline OmpDirectiveSet topDistributeSet{
    Directive::OMPD_distribute,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
};

const inline OmpDirectiveSet allDistributeSet{
    llvm::omp::OMPD_distribute,
    llvm::omp::OMPD_distribute_parallel_do,
    llvm::omp::OMPD_distribute_parallel_do_simd,
    llvm::omp::OMPD_distribute_simd,
    llvm::omp::OMPD_target_teams_distribute,
    llvm::omp::OMPD_target_teams_distribute_parallel_do,
    llvm::omp::OMPD_target_teams_distribute_parallel_do_simd,
    llvm::omp::OMPD_target_teams_distribute_simd,
    llvm::omp::OMPD_teams_distribute,
    llvm::omp::OMPD_teams_distribute_parallel_do,
    llvm::omp::OMPD_teams_distribute_parallel_do_simd,
    llvm::omp::OMPD_teams_distribute_simd,
};

//===----------------------------------------------------------------------===//
// Directive sets for groups of multiple directives
//===----------------------------------------------------------------------===//

const inline OmpDirectiveSet allDoSimdSet = allDoSet & allSimdSet;

const inline OmpDirectiveSet workShareSet{
    OmpDirectiveSet{
        Directive::OMPD_workshare,
        Directive::OMPD_parallel_workshare,
        Directive::OMPD_parallel_sections,
        Directive::OMPD_sections,
        Directive::OMPD_single,
    } | allDoSet,
};

const inline OmpDirectiveSet taskGeneratingSet{
    OmpDirectiveSet{
        Directive::OMPD_task,
    } | allTaskloopSet,
};

const inline OmpDirectiveSet nonPartialVarSet{
    Directive::OMPD_allocate,
    Directive::OMPD_allocators,
    Directive::OMPD_threadprivate,
    Directive::OMPD_declare_target,
};

//===----------------------------------------------------------------------===//
// Directive sets for allowed/not allowed nested directives
//===----------------------------------------------------------------------===//

const inline OmpDirectiveSet nestedOrderedErrSet{
    Directive::OMPD_critical,
    Directive::OMPD_ordered,
    Directive::OMPD_atomic,
    Directive::OMPD_task,
    Directive::OMPD_taskloop,
};

const inline OmpDirectiveSet nestedWorkshareErrSet{
    OmpDirectiveSet{
        Directive::OMPD_task,
        Directive::OMPD_taskloop,
        Directive::OMPD_critical,
        Directive::OMPD_ordered,
        Directive::OMPD_atomic,
        Directive::OMPD_master,
    } | workShareSet,
};

const inline OmpDirectiveSet nestedMasterErrSet{
    OmpDirectiveSet{
        Directive::OMPD_atomic,
    } | taskGeneratingSet |
        workShareSet,
};

const inline OmpDirectiveSet nestedBarrierErrSet{
    OmpDirectiveSet{
        Directive::OMPD_critical,
        Directive::OMPD_ordered,
        Directive::OMPD_atomic,
        Directive::OMPD_master,
    } | taskGeneratingSet |
        workShareSet,
};

const inline OmpDirectiveSet nestedTeamsAllowedSet{
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_master,
    Directive::OMPD_parallel_master_taskloop,
    Directive::OMPD_parallel_master_taskloop_simd,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
    Directive::OMPD_distribute,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
};

const inline OmpDirectiveSet nestedOrderedParallelErrSet{
    Directive::OMPD_parallel,
    Directive::OMPD_target_parallel,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
};

const inline OmpDirectiveSet nestedOrderedDoAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_parallel_do,
    Directive::OMPD_target_parallel_do,
};

const inline OmpDirectiveSet nestedCancelTaskgroupAllowedSet{
    Directive::OMPD_task,
    Directive::OMPD_taskloop,
};

const inline OmpDirectiveSet nestedCancelSectionsAllowedSet{
    Directive::OMPD_sections,
    Directive::OMPD_parallel_sections,
};

const inline OmpDirectiveSet nestedCancelDoAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_parallel_do,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do,
};

const inline OmpDirectiveSet nestedCancelParallelAllowedSet{
    Directive::OMPD_parallel,
    Directive::OMPD_target_parallel,
};

const inline OmpDirectiveSet nestedReduceWorkshareAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_sections,
    Directive::OMPD_do_simd,
};
} // namespace llvm::omp

#endif // FORTRAN_SEMANTICS_OPENMP_DIRECTIVE_SETS_H_
