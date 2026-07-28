//===-- include/flang/Semantics/openmp-directive-sets.h ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_OPENMP_DIRECTIVE_SETS_H_
#define FORTRAN_SEMANTICS_OPENMP_DIRECTIVE_SETS_H_

#include "llvm/Frontend/OpenMP/OMP.h"

namespace llvm::omp {
//===----------------------------------------------------------------------===//
// Directive sets for single directives
//===----------------------------------------------------------------------===//
// - top<Directive>Set: The directive appears alone or as the first in a
//   compound construct.
// - bottom<Directive>Set: The directive appears alone or as the last in a
//   compound construct.
// - all<Directive>Set: All standalone or compound uses of the directive.

static const llvm::omp::DirectiveSet topDistributeSet{
    Directive::OMPD_distribute,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
};

static const llvm::omp::DirectiveSet allDistributeSet{
    llvm::omp::DirectiveSet{
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

static const llvm::omp::DirectiveSet topDoSet{
    Directive::OMPD_do,
    Directive::OMPD_do_simd,
};

static const llvm::omp::DirectiveSet allDoSet{
    llvm::omp::DirectiveSet{
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

static const llvm::omp::DirectiveSet topLoopSet{
    Directive::OMPD_loop,
};

static const llvm::omp::DirectiveSet allLoopSet{
    llvm::omp::DirectiveSet{
        Directive::OMPD_parallel_loop,
        Directive::OMPD_target_parallel_loop,
        Directive::OMPD_target_teams_loop,
        Directive::OMPD_teams_loop,
    } | topLoopSet,
};

static const llvm::omp::DirectiveSet topParallelSet{
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_loop,
    Directive::OMPD_parallel_masked,
    Directive::OMPD_parallel_masked_taskloop,
    Directive::OMPD_parallel_masked_taskloop_simd,
    Directive::OMPD_parallel_master,
    Directive::OMPD_parallel_master_taskloop,
    Directive::OMPD_parallel_master_taskloop_simd,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
};

static const llvm::omp::DirectiveSet allParallelSet{
    llvm::omp::DirectiveSet{
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

static const llvm::omp::DirectiveSet topSimdSet{
    Directive::OMPD_simd,
};

static const llvm::omp::DirectiveSet allSimdSet{
    llvm::omp::DirectiveSet{
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

static const llvm::omp::DirectiveSet topTargetSet{
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

static const llvm::omp::DirectiveSet allTargetSet{topTargetSet};

static const llvm::omp::DirectiveSet topTaskloopSet{
    Directive::OMPD_taskloop,
    Directive::OMPD_taskloop_simd,
};

static const llvm::omp::DirectiveSet allTaskloopSet{
    llvm::omp::DirectiveSet{
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

static const llvm::omp::DirectiveSet topTeamsSet{
    Directive::OMPD_teams,
    Directive::OMPD_teams_distribute,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_simd,
    Directive::OMPD_teams_loop,
    Directive::OMPD_teams_workdistribute,
};

static const llvm::omp::DirectiveSet bottomTeamsSet{
    Directive::OMPD_target_teams,
    Directive::OMPD_teams,
};

static const llvm::omp::DirectiveSet allTeamsSet{
    llvm::omp::DirectiveSet{
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
static const llvm::omp::DirectiveSet allDistributeParallelDoSet{
    allDistributeSet & allParallelSet & allDoSet};
static const llvm::omp::DirectiveSet allDistributeParallelDoSimdSet{
    allDistributeSet & allParallelSet & allDoSet & allSimdSet};
static const llvm::omp::DirectiveSet allDistributeSimdSet{
    allDistributeSet & allSimdSet};
static const llvm::omp::DirectiveSet allDoSimdSet{allDoSet & allSimdSet};
static const llvm::omp::DirectiveSet allTaskloopSimdSet{
    allTaskloopSet & allSimdSet};

static const llvm::omp::DirectiveSet compositeConstructSet{
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
    Directive::OMPD_do_simd,
    Directive::OMPD_taskloop_simd,
};

static const llvm::omp::DirectiveSet blockConstructSet{
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

static const llvm::omp::DirectiveSet loopConstructSet{
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
    Directive::OMPD_fuse,
    Directive::OMPD_tile,
    Directive::OMPD_unroll,
    Directive::OMPD_interchange,
};

static const llvm::omp::DirectiveSet loopTransformationSet{
    Directive::OMPD_tile,
    Directive::OMPD_unroll,
    Directive::OMPD_fuse,
    Directive::OMPD_interchange,
};

static const llvm::omp::DirectiveSet nonPartialVarSet{
    Directive::OMPD_allocate,
    Directive::OMPD_allocators,
    Directive::OMPD_threadprivate,
    Directive::OMPD_declare_target,
};

static const llvm::omp::DirectiveSet taskGeneratingSet{
    llvm::omp::DirectiveSet{
        Directive::OMPD_task,
    } | allTaskloopSet,
};

static const llvm::omp::DirectiveSet workShareSet{
    llvm::omp::DirectiveSet{
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

static const llvm::omp::DirectiveSet scanParentAllowedSet{
    allDoSet | allSimdSet};

//===----------------------------------------------------------------------===//
// Directive sets for allowed/not allowed nested directives
//===----------------------------------------------------------------------===//

static const llvm::omp::DirectiveSet nestedBarrierErrSet{
    llvm::omp::DirectiveSet{
        Directive::OMPD_atomic,
        Directive::OMPD_critical,
        Directive::OMPD_master,
        Directive::OMPD_ordered,
    } | taskGeneratingSet |
        workShareSet,
};

static const llvm::omp::DirectiveSet nestedCancelDoAllowedSet{
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_do,
    Directive::OMPD_parallel_do,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do,
};

static const llvm::omp::DirectiveSet nestedCancelParallelAllowedSet{
    Directive::OMPD_parallel,
    Directive::OMPD_target_parallel,
};

static const llvm::omp::DirectiveSet nestedCancelSectionsAllowedSet{
    Directive::OMPD_parallel_sections,
    Directive::OMPD_sections,
};

static const llvm::omp::DirectiveSet nestedCancelTaskgroupAllowedSet{
    Directive::OMPD_task,
    Directive::OMPD_taskloop,
};

static const llvm::omp::DirectiveSet nestedMasterErrSet{
    llvm::omp::DirectiveSet{
        Directive::OMPD_atomic,
    } | taskGeneratingSet |
        workShareSet,
};

static const llvm::omp::DirectiveSet nestedOrderedDoAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_parallel_do,
    Directive::OMPD_target_parallel_do,
};

static const llvm::omp::DirectiveSet nestedOrderedErrSet{
    Directive::OMPD_atomic,
    Directive::OMPD_critical,
    Directive::OMPD_ordered,
    Directive::OMPD_task,
    Directive::OMPD_taskloop,
};

static const llvm::omp::DirectiveSet nestedOrderedParallelErrSet{
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
    Directive::OMPD_target_parallel,
};

static const llvm::omp::DirectiveSet nestedReduceWorkshareAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_do_simd,
    Directive::OMPD_sections,
};

static const llvm::omp::DirectiveSet nestedTeamsAllowedSet{
    Directive::OMPD_workdistribute,
    Directive::OMPD_distribute,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
    Directive::OMPD_loop,
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_loop,
    Directive::OMPD_parallel_master,
    Directive::OMPD_parallel_master_taskloop,
    Directive::OMPD_parallel_master_taskloop_simd,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
};

static const llvm::omp::DirectiveSet nestedWorkshareErrSet{
    llvm::omp::DirectiveSet{
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
static const llvm::omp::DirectiveSet simpleStandaloneNonSimdOnlySet{
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
