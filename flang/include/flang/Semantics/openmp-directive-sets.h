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
//   compound construct.
// - bottom<Directive>Set: The directive appears alone or as the last in a
//   compound construct.
// - all<Directive>Set: All standalone or compound uses of the directive.

static const OmpDirectiveSet topDistributeSet{
    Directive::OMPD_distribute,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
};

static const OmpDirectiveSet allDistributeSet{
    OmpDirectiveSet{
        Directive::OMPD_target_teams_distribute,
        Directive::OMPD_target_teams_distribute_parallel_do,
        Directive::OMPD_target_teams_distribute_parallel_do_simd,
        Directive::OMPD_target_teams_distribute_simd,
        Directive::OMPD_teams_distribute,
        Directive::OMPD_teams_distribute_parallel_do,
        Directive::OMPD_teams_distribute_parallel_do_simd,
        Directive::OMPD_teams_distribute_simd,
    } | topDistributeSet,
};

static const OmpDirectiveSet topDoSet{
    Directive::OMPD_do,
    Directive::OMPD_do_simd,
};

static const OmpDirectiveSet allDoSet{
    OmpDirectiveSet{
        Directive::OMPD_distribute_parallel_do,
        Directive::OMPD_distribute_parallel_do_simd,
        Directive::OMPD_parallel_do,
        Directive::OMPD_parallel_do_simd,
        Directive::OMPD_target_parallel_do,
        Directive::OMPD_target_parallel_do_simd,
        Directive::OMPD_target_teams_distribute_parallel_do,
        Directive::OMPD_target_teams_distribute_parallel_do_simd,
        Directive::OMPD_teams_distribute_parallel_do,
        Directive::OMPD_teams_distribute_parallel_do_simd,
    } | topDoSet,
};

static const OmpDirectiveSet topLoopSet{
    Directive::OMPD_loop,
};

static const OmpDirectiveSet allLoopSet{
    OmpDirectiveSet{
        Directive::OMPD_parallel_loop,
        Directive::OMPD_target_parallel_loop,
        Directive::OMPD_target_teams_loop,
        Directive::OMPD_teams_loop,
    } | topLoopSet,
};

static const OmpDirectiveSet topParallelSet{
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_loop,
    Directive::OMPD_parallel_masked_taskloop,
    Directive::OMPD_parallel_masked_taskloop_simd,
    Directive::OMPD_parallel_master_taskloop,
    Directive::OMPD_parallel_master_taskloop_simd,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
};

static const OmpDirectiveSet allParallelSet{
    OmpDirectiveSet{
        Directive::OMPD_distribute_parallel_do,
        Directive::OMPD_distribute_parallel_do_simd,
        Directive::OMPD_target_parallel,
        Directive::OMPD_target_parallel_do,
        Directive::OMPD_target_parallel_do_simd,
        Directive::OMPD_target_parallel_loop,
        Directive::OMPD_target_teams_distribute_parallel_do,
        Directive::OMPD_target_teams_distribute_parallel_do_simd,
        Directive::OMPD_teams_distribute_parallel_do,
        Directive::OMPD_teams_distribute_parallel_do_simd,
    } | topParallelSet,
};

static const OmpDirectiveSet topSimdSet{
    Directive::OMPD_simd,
};

static const OmpDirectiveSet allSimdSet{
    OmpDirectiveSet{
        Directive::OMPD_distribute_parallel_do_simd,
        Directive::OMPD_distribute_simd,
        Directive::OMPD_do_simd,
        Directive::OMPD_masked_taskloop_simd,
        Directive::OMPD_master_taskloop_simd,
        Directive::OMPD_parallel_do_simd,
        Directive::OMPD_parallel_masked_taskloop_simd,
        Directive::OMPD_parallel_master_taskloop_simd,
        Directive::OMPD_target_parallel_do_simd,
        Directive::OMPD_target_simd,
        Directive::OMPD_target_teams_distribute_parallel_do_simd,
        Directive::OMPD_target_teams_distribute_simd,
        Directive::OMPD_taskloop_simd,
        Directive::OMPD_teams_distribute_parallel_do_simd,
        Directive::OMPD_teams_distribute_simd,
    } | topSimdSet,
};

static const OmpDirectiveSet topTargetSet{
    Directive::OMPD_target,
    Directive::OMPD_target_parallel,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_parallel_loop,
    Directive::OMPD_target_simd,
    Directive::OMPD_target_teams,
    Directive::OMPD_target_teams_distribute,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_simd,
    Directive::OMPD_target_teams_loop,
    Directive::OMPD_target_teams_workdistribute,
};

static const OmpDirectiveSet allTargetSet{topTargetSet};

static const OmpDirectiveSet topTaskloopSet{
    Directive::OMPD_taskloop,
    Directive::OMPD_taskloop_simd,
};

static const OmpDirectiveSet allTaskloopSet{
    OmpDirectiveSet{
        Directive::OMPD_masked_taskloop,
        Directive::OMPD_masked_taskloop_simd,
        Directive::OMPD_master_taskloop,
        Directive::OMPD_master_taskloop_simd,
        Directive::OMPD_parallel_masked_taskloop,
        Directive::OMPD_parallel_masked_taskloop_simd,
        Directive::OMPD_parallel_master_taskloop,
        Directive::OMPD_parallel_master_taskloop_simd,
    } | topTaskloopSet,
};

static const OmpDirectiveSet topTeamsSet{
    Directive::OMPD_teams,
    Directive::OMPD_teams_distribute,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_simd,
    Directive::OMPD_teams_loop,
    Directive::OMPD_teams_workdistribute,
};

static const OmpDirectiveSet bottomTeamsSet{
    Directive::OMPD_target_teams,
    Directive::OMPD_teams,
};

static const OmpDirectiveSet allTeamsSet{
    OmpDirectiveSet{
        Directive::OMPD_target_teams,
        Directive::OMPD_target_teams_distribute,
        Directive::OMPD_target_teams_distribute_parallel_do,
        Directive::OMPD_target_teams_distribute_parallel_do_simd,
        Directive::OMPD_target_teams_distribute_simd,
        Directive::OMPD_target_teams_loop,
        Directive::OMPD_target_teams_workdistribute,
    } | topTeamsSet,
};

//===----------------------------------------------------------------------===//
// Directive sets for groups of multiple directives
//===----------------------------------------------------------------------===//

// Composite constructs
static const OmpDirectiveSet allDistributeParallelDoSet{
    allDistributeSet & allParallelSet & allDoSet};
static const OmpDirectiveSet allDistributeParallelDoSimdSet{
    allDistributeSet & allParallelSet & allDoSet & allSimdSet};
static const OmpDirectiveSet allDistributeSimdSet{
    allDistributeSet & allSimdSet};
static const OmpDirectiveSet allDoSimdSet{allDoSet & allSimdSet};
static const OmpDirectiveSet allTaskloopSimdSet{allTaskloopSet & allSimdSet};

static const OmpDirectiveSet compositeConstructSet{
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
    Directive::OMPD_do_simd,
    Directive::OMPD_taskloop_simd,
};

static const OmpDirectiveSet blockConstructSet{
    Directive::OMPD_masked,
    Directive::OMPD_master,
    Directive::OMPD_ordered,
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_masked,
    Directive::OMPD_parallel_master,
    Directive::OMPD_parallel_workshare,
    Directive::OMPD_scope,
    Directive::OMPD_single,
    Directive::OMPD_target,
    Directive::OMPD_target_data,
    Directive::OMPD_target_parallel,
    Directive::OMPD_target_teams,
    Directive::OMPD_task,
    Directive::OMPD_taskgroup,
    Directive::OMPD_teams,
    Directive::OMPD_workshare,
    Directive::OMPD_target_teams_workdistribute,
    Directive::OMPD_teams_workdistribute,
    Directive::OMPD_workdistribute,
};

static const OmpDirectiveSet loopConstructSet{
    Directive::OMPD_distribute,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
    Directive::OMPD_do,
    Directive::OMPD_do_simd,
    Directive::OMPD_loop,
    Directive::OMPD_masked_taskloop,
    Directive::OMPD_masked_taskloop_simd,
    Directive::OMPD_master_taskloop,
    Directive::OMPD_master_taskloop_simd,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_loop,
    Directive::OMPD_parallel_masked_taskloop,
    Directive::OMPD_parallel_masked_taskloop_simd,
    Directive::OMPD_parallel_master_taskloop,
    Directive::OMPD_parallel_master_taskloop_simd,
    Directive::OMPD_simd,
    Directive::OMPD_target_loop,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_parallel_loop,
    Directive::OMPD_target_simd,
    Directive::OMPD_target_teams_distribute,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_simd,
    Directive::OMPD_target_teams_loop,
    Directive::OMPD_taskloop,
    Directive::OMPD_taskloop_simd,
    Directive::OMPD_teams_distribute,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_simd,
    Directive::OMPD_teams_loop,
    Directive::OMPD_tile,
    Directive::OMPD_unroll,
};

static const OmpDirectiveSet nonPartialVarSet{
    Directive::OMPD_allocate,
    Directive::OMPD_allocators,
    Directive::OMPD_threadprivate,
    Directive::OMPD_declare_target,
};

static const OmpDirectiveSet taskGeneratingSet{
    OmpDirectiveSet{
        Directive::OMPD_task,
    } | allTaskloopSet,
};

static const OmpDirectiveSet workShareSet{
    OmpDirectiveSet{
        Directive::OMPD_workshare,
        Directive::OMPD_parallel_workshare,
        Directive::OMPD_parallel_sections,
        Directive::OMPD_scope,
        Directive::OMPD_sections,
        Directive::OMPD_single,
    } | allDoSet,
};

//===----------------------------------------------------------------------===//
// Directive sets for parent directives that do allow/not allow a construct
//===----------------------------------------------------------------------===//

static const OmpDirectiveSet scanParentAllowedSet{allDoSet | allSimdSet};

//===----------------------------------------------------------------------===//
// Directive sets for allowed/not allowed nested directives
//===----------------------------------------------------------------------===//

static const OmpDirectiveSet nestedBarrierErrSet{
    OmpDirectiveSet{
        Directive::OMPD_atomic,
        Directive::OMPD_critical,
        Directive::OMPD_master,
        Directive::OMPD_ordered,
    } | taskGeneratingSet |
        workShareSet,
};

static const OmpDirectiveSet nestedCancelDoAllowedSet{
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_do,
    Directive::OMPD_parallel_do,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do,
};

static const OmpDirectiveSet nestedCancelParallelAllowedSet{
    Directive::OMPD_parallel,
    Directive::OMPD_target_parallel,
};

static const OmpDirectiveSet nestedCancelSectionsAllowedSet{
    Directive::OMPD_parallel_sections,
    Directive::OMPD_sections,
};

static const OmpDirectiveSet nestedCancelTaskgroupAllowedSet{
    Directive::OMPD_task,
    Directive::OMPD_taskloop,
};

static const OmpDirectiveSet nestedMasterErrSet{
    OmpDirectiveSet{
        Directive::OMPD_atomic,
    } | taskGeneratingSet |
        workShareSet,
};

static const OmpDirectiveSet nestedOrderedDoAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_parallel_do,
    Directive::OMPD_target_parallel_do,
};

static const OmpDirectiveSet nestedOrderedErrSet{
    Directive::OMPD_atomic,
    Directive::OMPD_critical,
    Directive::OMPD_ordered,
    Directive::OMPD_task,
    Directive::OMPD_taskloop,
};

static const OmpDirectiveSet nestedOrderedParallelErrSet{
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
    Directive::OMPD_target_parallel,
};

static const OmpDirectiveSet nestedReduceWorkshareAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_do_simd,
    Directive::OMPD_sections,
};

static const OmpDirectiveSet nestedTeamsAllowedSet{
    Directive::OMPD_workdistribute,
    Directive::OMPD_distribute,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
    Directive::OMPD_loop,
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_master,
    Directive::OMPD_parallel_master_taskloop,
    Directive::OMPD_parallel_master_taskloop_simd,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
};

static const OmpDirectiveSet nestedWorkshareErrSet{
    OmpDirectiveSet{
        Directive::OMPD_atomic,
        Directive::OMPD_critical,
        Directive::OMPD_master,
        Directive::OMPD_ordered,
        Directive::OMPD_task,
        Directive::OMPD_taskloop,
    } | workShareSet,
};

//===----------------------------------------------------------------------===//
// Misc directive sets
//===----------------------------------------------------------------------===//

// Simple standalone directives than can be erased by -fopenmp-simd.
static const OmpDirectiveSet simpleStandaloneNonSimdOnlySet{
    Directive::OMPD_taskyield,
    Directive::OMPD_barrier,
    Directive::OMPD_ordered,
    Directive::OMPD_target_enter_data,
    Directive::OMPD_target_exit_data,
    Directive::OMPD_target_update,
    Directive::OMPD_taskwait,
};

} // namespace llvm::omp

#endif // FORTRAN_SEMANTICS_OPENMP_DIRECTIVE_SETS_H_
