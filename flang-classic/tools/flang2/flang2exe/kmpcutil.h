/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef KMPC_RUNTIME_H_
#define KMPC_RUNTIME_H_

#include "gbldefs.h"
#include "global.h"
#include "symtab.h"
#include "ili.h"

/** \file
 * \brief Various definitions for the kmpc runtime
 */

/* KMPC Task Flags
 * See KMPC's kmp.h struct kmp_tasking_flags
 */
#define KMPC_TASK_UNTIED 0x00
#define KMPC_TASK_TIED 0x01
#define KMPC_TASK_FINAL 0x02
#define KMPC_TASK_MERGED_IF0 0x04
#define KMPC_TASK_DTOR_THK 0x08
#define KMPC_TASK_PROXY 0x10
#define KMPC_TASK_PRIORITY 0x20

/* KMPC Schedule Types
 * https://www.openmprtl.org/sites/default/files/resources/libomp_20151009_manual.pdf
 * Additional types mentioned in the source and used in the refereced manual's
 * example (KMP_SCH_DYNAMIC_CHUNKED).
 */
typedef enum _kmpc_sched_e {
  KMP_SCH_NONE = 0,
  KMP_SCH_LOWER = 32,
  KMP_SCH_STATIC_CHUNKED = 33,
  KMP_SCH_STATIC = 34,
  KMP_SCH_DYNAMIC_CHUNKED = 35,
  KMP_SCH_GUIDED_CHUNKED = 36,
  KMP_SCH_RUNTIME_CHUNKED = 37,
  KMP_SCH_AUTO = 38,
  KMP_SCH_STATIC_STEAL = 44,
  KMP_SCH_UPPER = 45,
  KMP_ORD_LOWER = 64,
  KMP_ORD_STATIC_CHUNKED = 65,
  KMP_ORD_STATIC = 66,
  KMP_ORD_DYNAMIC_CHUNKED = 67,
  KMP_ORD_GUIDED_CHUNKED = 68,
  KMP_ORD_RUNTIME = 69,
  KMP_ORD_AUTO = 70,
  KMP_ORD_UPPER = 72,
  KMP_DISTRIBUTE_STATIC_CHUNKED = 91,
  KMP_DISTRIBUTE_STATIC = 92,
  KMP_DISTRIBUTE_STATIC_CHUNKED_CHUNKONE = 93,
  KMP_NM_LOWER = 160,
  KMP_NM_STATIC = 162,
  KMP_NM_GUIDED_CHUNKED = 164,
  KMP_NM_AUTO = 166,
  KMP_NM_ORD_STATIC = 194,
  KMP_NM_ORD_AUTO = 198,
  KMP_NM_UPPER = 200,
  KMP_SCH_DEFAULT = KMP_SCH_STATIC
} kmpc_sched_e;

typedef enum RegionType { OPENMP, OPENACC } RegionType;

/* Argument type used for handling for loops and scheduling.
 * All values here are sptrs.
 */
typedef struct _loop_args_t {
  SPTR lower;
  SPTR upper;
  SPTR stride;
  SPTR chunk;
  SPTR last;
  SPTR upperd;
  DTYPE dtype;        /* Lower/Upper bound data type INT,INT8,UINT, UINT8 */
  kmpc_sched_e sched; /* KMPC schedule type */
} loop_args_t;

struct kmpc_api_entry_t {
  const char *name;      /* KMPC API function name                    */
  const ILI_OP ret_iliopc;  /* KMPC API function return value ili opcode */
  const DTYPE ret_dtype; /* KMPC API function return value type       */
  const int flags;       /* (Optional) See KMPC_FLAG_XXX above        */
};

/* Used internally for creating structs, or representing formal parameters when
 * generating fortran outlined function/task signatures.
 */
typedef struct any_kmpc_struct {
  char *name;
  DTYPE dtype;
  int byval;
  int psptr;
} KMPC_ST_TYPE;

/* KMPC API macros and structs */
enum {
  KMPC_API_BAD,
  KMPC_API_FORK_CALL,
  KMPC_API_BARRIER,
  KMPC_API_CANCEL_BARRIER,
  KMPC_API_COPYPRIVATE,
  KMPC_API_CRITICAL,
  KMPC_API_END_CRITICAL,
  KMPC_API_SINGLE,
  KMPC_API_END_SINGLE,
  KMPC_API_MASTER,
  KMPC_API_END_MASTER,
  KMPC_API_FLUSH,
  KMPC_API_ORDERED,
  KMPC_API_END_ORDERED,
  KMPC_API_FOR_STATIC_INIT,
  KMPC_API_FOR_STATIC_FINI,
  KMPC_API_DISPATCH_INIT,
  KMPC_API_DISPATCH_NEXT,
  KMPC_API_DISPATCH_FINI,
  KMPC_API_GLOBAL_THREAD_NUM,
  KMPC_API_GLOBAL_NUM_THREADS,
  KMPC_API_BOUND_THREAD_NUM,
  KMPC_API_BOUND_NUM_THREADS,
  KMPC_API_PUSH_NUM_THREADS,
  KMPC_API_SERIALIZED_PARALLEL,
  KMPC_API_END_SERIALIZED_PARALLEL,
  KMPC_API_THREADPRIVATE_CACHED,
  KMPC_API_THREADPRIVATE_REGISTER_VEC,
  KMPC_API_THREADPRIVATE_REGISTER,
  KMPC_API_TASK_ALLOC,
  KMPC_API_TASK,
  KMPC_API_TASK_BEGIN_IF0,
  KMPC_API_TASK_COMPLETE_IF0,
  KMPC_API_TASK_WAIT,
  KMPC_API_TASK_YIELD,
  KMPC_API_CANCEL,
  KMPC_API_CANCELLATIONPOINT,
  KMPC_API_TASKGROUP,
  KMPC_API_END_TASKGROUP,
  KMPC_API_TASK_WITH_DEPS,
  KMPC_API_WAIT_DEPS,
  KMPC_API_TASKLOOP,
  KMPC_API_THREADPRIVATE,
  KMPC_API_PUSH_NUM_TEAMS,
  KMPC_API_FORK_TEAMS,
  KMPC_API_DIST_FOR_STATIC_INIT,
  KMPC_API_DIST_DISPATCH_INIT,
  KMPC_API_PUSH_PROC_BIND,
  KMPC_API_ATOMIC_RD,
  KMPC_API_ATOMIC_WR,
  /* Begin - OpenMP Accelerator RT (libomptarget-nvptx) - non standard - */
  KMPC_API_PUSH_TARGET_TRIPCOUNT,
  KMPC_API_FOR_STATIC_INIT_SIMPLE_SPMD,
  KMPC_API_SPMD_KERNEL_INIT,
  KMPC_API_KERNEL_INIT_PARAMS,
  KMPC_API_SHUFFLE_I32,
  KMPC_API_SHUFFLE_I64,
  KMPC_API_NVPTX_PARALLEL_REDUCE_NOWAIT_SIMPLE_SPMD,
  KMPC_API_NVPTX_END_REDUCE_NOWAIT,
  /* End - OpenMP Accelerator RT (libomptarget-nvptx) - non standard - */
  KMPC_API_N_ENTRIES /* <-- Always last */
};

/**
   \brief ...
 */
int ll_make_kmpc_atomic_read(int *opnd, DTYPE dtype);

/**
   \brief ...
 */
int ll_make_kmpc_atomic_write(int *opnd, DTYPE dtype);

/**
   \brief ...
 */
int ll_make_kmpc_barrier(void);

/**
   \brief ...
 */
int ll_make_kmpc_bound_num_threads(void);

/**
   \brief ...
 */
int ll_make_kmpc_bound_thread_num(void);

/**
   \brief ...
 */
int ll_make_kmpc_cancel_barrier(void);

/**
   \brief ...
 */
int ll_make_kmpc_cancel(int argili);

/**
   \brief ...
 */
int ll_make_kmpc_cancellationpoint(int argili);

/// Return a result or JSR ili to __kmpc_copyprivate()
int ll_make_kmpc_copyprivate(SPTR array_sptr, int single_ili,
                             int copyfunc_acon);

/// Return a result or JSR ili to __kmpc_critical()
int ll_make_kmpc_critical(SPTR sem);

/**
   \brief ...
 */
int ll_make_kmpc_dispatch_fini(DTYPE dtype);

/**
   \brief ...
 */
int ll_make_kmpc_dispatch_init(const loop_args_t *inargs);

/// Return a result or JSR ili to __kmpc_dispatch_next_<size><signed|unsigned>
/// lower, upper, stride: sptrs
int ll_make_kmpc_dispatch_next(SPTR lower, SPTR upper, SPTR stride, SPTR last,
                               DTYPE dtype);

/**
   \brief ...
 */
int ll_make_kmpc_dist_dispatch_init(const loop_args_t *inargs);

/**
   \brief ...
 */
int ll_make_kmpc_dist_for_static_init(const loop_args_t *inargs);

/// Return a result or JSR ili to __kmpc_end_critical()
int ll_make_kmpc_end_critical(SPTR sem);

/**
   \brief ...
 */
int ll_make_kmpc_end_master(void);

/**
   \brief ...
 */
int ll_make_kmpc_end_ordered(void);

/**
   \brief ...
 */
int ll_make_kmpc_end_serialized_parallel(void);

/**
   \brief ...
 */
int ll_make_kmpc_end_single(void);

/**
   \brief ...
 */
int ll_make_kmpc_end_taskgroup(void);

/// Return a result or JSR ili to __kmpc_flush()
int ll_make_kmpc_flush(void);

/**
   \brief ...
 */
int ll_make_kmpc_fork_call(SPTR sptr, int argc, int *arglist, RegionType rt, int ngangs_ili);

/**
   \brief ...
 */
int ll_make_kmpc_fork_teams(SPTR sptr, int argc, int *arglist);

/**
   \brief ...
 */
int ll_make_kmpc_for_static_fini(void);

/**
   \brief ...
 */
int ll_make_kmpc_for_static_init_args(DTYPE dtype, int *inargs);

/**
   \brief ...
 */
int ll_make_kmpc_for_static_init(const loop_args_t *inargs);

/**
   \brief ...
 */
int ll_make_kmpc_global_num_threads(void);

/**
   \brief ...
 */
int ll_make_kmpc_global_thread_num(void);

/**
   \brief ...
 */
int ll_make_kmpc_master(void);

/**
   \brief ...
 */
int ll_make_kmpc_omp_wait_deps(const loop_args_t *inargs);

/**
   \brief ...
 */
int ll_make_kmpc_ordered(void);

/**
   \brief ...
 */
int ll_make_kmpc_push_num_teams(int nteams_ili, int thread_limit_ili);

/**
   \brief ...
 */
int ll_make_kmpc_push_num_threads(int argili);

/**
   \brief ...
 */
int ll_make_kmpc_push_proc_bind(int argili);

/**
   \brief ...
 */
int ll_make_kmpc_serialized_parallel(void);

/**
   \brief ...
 */
int ll_make_kmpc_single(void);

/**
   \brief ...
 */
DTYPE ll_make_kmpc_struct_type(int count, const char *name,
                               KMPC_ST_TYPE *meminfo, ISZ_T sz);

/// Return an sptr to the allocated task object:  __kmp_omp_task_alloc()
/// \param base  sptr for storing return value from __kmpc_omp_task_alloc
/// \param sptr  sptr representing the outlined function that is the task
/// \param scope_sptr ST_BLOCK containing the uplevel block
/// \param flags MP_TASK_xxx flags (see mp.h)
SPTR ll_make_kmpc_task_arg(SPTR base, SPTR sptr, SPTR scope_sptr,
                           SPTR flags_sptr);

/// Return a JSR ili to __kmpc_omp_task_begin_if0.
/// \param task_sptr sptr representing the allocated task
int ll_make_kmpc_task_begin_if0(SPTR task_sptr);

/// Return a JSR ili to __kmpc_omp_task_complete_if0.
/// \param task_sptr sptr representing the allocated task
int ll_make_kmpc_task_complete_if0(SPTR task_sptr);

/**
   \brief ...
 */
int ll_make_kmpc_taskgroup(void);

/**
   \brief ...
 */
int ll_make_kmpc_task(SPTR task_sptr);

/**
   \brief ...
 */
int ll_make_kmpc_taskloop(int *inargs);

/**
   \brief ...
 */
int ll_make_kmpc_task_wait(void);

/**
   \brief ...
 */
int ll_make_kmpc_task_with_deps(const loop_args_t *inargs);

/**
   \brief ...
 */
int ll_make_kmpc_task_yield(void);

/**
   \brief ...
 */
int ll_make_kmpc_threadprivate_cached(int data_ili, int size_ili,
                                      int cache_ili);

/**
   \brief ...
 */
int ll_make_kmpc_threadprivate(int data_ili, int size_ili);

/**
   \brief ...
 */
int ll_make_kmpc_threadprivate_register(int data_ili, int ctor_ili,
                                        int cctor_ili, int dtor_ili);

/**
   \brief ...
 */
int ll_make_kmpc_threadprivate_register_vec(int data_ili, int ctor_ili,
                                            int cctor_ili, int dtor_ili,
                                            int size_ili);

/**
   \brief ...
 */
int mp_to_kmpc_tasking_flags(const int mp);

/**
   \brief Given a MP_ or DI_ schedule type and return the KMPC equivalent
 */
kmpc_sched_e mp_sched_to_kmpc_sched(int sched);

/**
   \brief ...
 */
void reset_kmpc_ident_dtype(void);

/* OpenMP Accelerator RT - non standard */
/* Only Available for linomptarget-nvptx device runtime */
#if defined(OMP_OFFLOAD_LLVM) || defined(OMP_OFFLOAD_PGI)
/**
   \brief cuda special register shuffling for int 32 or int 64
 */
int ll_make_kmpc_shuffle(int, int, int, bool);

/**
  \brief SPMD mode - static loop init
*/
int ll_make_kmpc_for_static_init_simple_spmd(const loop_args_t *, int);

/**
  \brief SPMD mode - kernel init.
*/
int ll_make_kmpc_spmd_kernel_init(int);

/**
  \brief Push the trip count of the loop that is going to be parallelized.
*/
int ll_make_kmpc_push_target_tripcount(int, SPTR);

/**
  \brief Parallel reduction within kernel for SPMD mode
*/
int ll_make_kmpc_nvptx_parallel_reduce_nowait_simple_spmd(int, int, int, SPTR, SPTR);

/**
  \brief End of reduction within kernel
*/
int ll_make_kmpc_nvptx_end_reduce_nowait();

/* End OpenMP Accelerator RT - non standard */
#endif
#endif /* KMPC_RUNTIME_H_ */
