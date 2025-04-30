/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

/*
 *  ompstubs.c:
 *
 *  libpgc stubs for omp_ routines not appearing in crit.o & omp.o for
 *  non-x64/x86 targets.
 *
 */

#include <stdio.h>
#include "komp.h"

int
omp_get_thread_limit(void)
{
  return 1;
}

int
omp_get_thread_limit_(void)
{
  return 1;
}

int
omp_in_parallel(void)
{
  return 0;
}

int
omp_in_parallel_(void)
{
  return 0;
}

void
omp_set_num_threads(int num_threads)
{
}

void
omp_set_num_threads_(int *num_threads)
{
}

void
omp_set_lock(omp_lock_t *sem)
{
}

void
omp_set_lock_(omp_lock_t *sem)
{
}

void
omp_unset_lock(omp_lock_t *sem)
{
}

void
omp_unset_lock_(omp_lock_t *sem)
{
}

int
omp_get_num_procs(void)
{
  return 1;
}

int
omp_get_num_procs_(void)
{
  return 1;
}

int
omp_get_num_threads(void)
{
  return 1;
}

int
omp_get_num_threads_(void)
{
  return 1;
}

int
omp_get_max_threads(void)
{
  return 1;
}

int
omp_get_max_threads_(void)
{
  return 1;
}

int
omp_get_thread_num(void)
{
  return 0;
}
int
omp_get_thread_num_(void)
{
  return 0;
}
void
omp_set_dynamic(int dynamic_threads)
{
}

void
omp_set_dynamic_(int *dynamic_threads)
{
}

int
omp_get_dynamic(void)
{
  return 0;
}

int
omp_get_dynamic_(void)
{
  return 0;
}

void
omp_set_nested(int nested)
{
}

void
omp_set_nested_(int *nested)
{
}

int
omp_get_nested(void)
{
  return 0;
}

int
omp_get_nested_(void)
{
  return 0;
}

void
omp_set_schedule(omp_sched_t kind, int modifier)
{
}

void
omp_set_schedule_(omp_sched_t *kind, int *modifier)
{
}

void
omp_get_schedule(omp_sched_t *kind, int *modifier)
{
  *kind = omp_sched_static;
  *modifier = 0;
}

void
omp_get_schedule_(omp_sched_t *kind, int *modifier)
{
  *kind = omp_sched_static;
  *modifier = 0;
}

void
omp_set_max_active_levels(int max_active_levels)
{
}

void
omp_set_max_active_levels_(int *max_active_levels)
{
}

int
omp_get_max_active_levels(void)
{
  return 0;
}

int
omp_get_max_active_levels_(void)
{
  return 0;
}

int
omp_get_level(void)
{
  return 0;
}

int
omp_get_level_(void)
{
  return 0;
}

int
omp_get_ancestor_thread_num(int level)
{
  if (level == 0) {
    return 0;
  }
  return -1;
}

int
omp_get_ancestor_thread_num_(int *level)
{
  if (*level == 0) {
    return 0;
  }
  return -1;
}

int
omp_get_team_size(int level)
{
  if (level == 0) {
    return 1;
  }
  return -1;
}

int
omp_get_team_size_(int *level)
{
  if (*level == 0) {
    return 1;
  }
  return -1;
}

int
omp_get_active_level(void)
{
  return 0;
}

int
omp_get_active_level_(void)
{
  return 0;
}

void
omp_init_lock(omp_lock_t *s)
{
}

void
omp_init_lock_(omp_lock_t *s)
{
}

void
omp_destroy_lock(omp_lock_t *arg)
{
}

void
omp_destroy_lock_(omp_lock_t *arg)
{
}

int
omp_test_lock(omp_lock_t *arg)
{
  return 0;
}

int
omp_test_lock_(omp_lock_t *arg)
{
  return 0;
}

void
omp_init_nest_lock(omp_nest_lock_t *arg)
{
}

void
omp_init_nest_lock_(omp_nest_lock_t *arg)
{
}

void
omp_destroy_nest_lock(omp_nest_lock_t *arg)
{
}

void
omp_destroy_nest_lock_(omp_nest_lock_t *arg)
{
}

void
omp_set_nest_lock(omp_nest_lock_t *arg)
{
}

void
omp_set_nest_lock_(omp_nest_lock_t *arg)
{
}

void
omp_unset_nest_lock(omp_nest_lock_t *arg)
{
}

void
omp_unset_nest_lock_(omp_nest_lock_t *arg)
{
}

int
omp_test_nest_lock(omp_nest_lock_t *arg)
{
  return 0;
}

int
omp_test_nest_lock_(omp_nest_lock_t *arg)
{
  return 0;
}

int
omp_get_cancellation()
{
  return 0;
}

int
omp_get_cancellation_()
{
  return 0;
}

omp_proc_bind_t 
omp_get_proc_bind_()
{
  return 0;
}

omp_proc_bind_t 
omp_get_proc_bind()
{
  return 0;
}

int 
omp_get_num_places()
{
  return 0;
}

int 
omp_get_num_places_()
{
  return 0;
}

int 
omp_get_place_num_procs(int placenum)
{
  return 0;
}

int 
omp_get_place_num_procs_(int placenum)
{
  return 0;
}
void 
omp_get_place_proc_ids(int place_num, int *ids)
{
  return;
}

void 
omp_get_place_proc_ids_(int place_num, int *ids)
{
  return;
}

int
omp_get_place_num()
{
  return -1;
}

int
omp_get_place_num_()
{
  return -1;
}

int 
omp_get_partition_num_places()
{
  return 0;
}

int 
omp_get_partition_num_places_()
{
  return 0;
}

void 
omp_get_partition_place_nums(int *place_nums)
{
}

void 
omp_get_partition_place_nums_(int *place_nums)
{
}

void 
omp_set_default_device(int device_num)
{
}

void 
omp_set_default_device_(int device_num)
{
}

int 
omp_get_default_device(void)
{
  return 0;
}

int 
omp_get_default_device_(void)
{
  return 0;
}

int 
omp_get_num_devices(void)
{
  return 0;
}

int 
omp_get_num_devices_(void)
{
  return 0;
}

int 
omp_get_num_teams(void)
{
  return 1;
}

int 
omp_get_num_teams_(void)
{
  return 1;
}

int 
omp_get_team_num(void)
{
  return 0;
}

int 
omp_get_team_num_(void)
{
  return 0;
}

int 
omp_is_initial_device(void)
{
  return 1;
}

int 
omp_is_initial_device_(void)
{
  return 1;
}

int 
omp_get_initial_device(void)
{
  return -10;
}

int 
omp_get_initial_device_(void)
{
  return -10;
}

int 
omp_get_max_task_priority(void)
{
  return 0;
}

int 
omp_get_max_task_priority_(void)
{
  return 0;
}

void 
omp_init_nest_lock_with_hint(omp_nest_lock_t *arg, omp_lock_hint_t hint)
{
  omp_init_nest_lock(arg);
}

void 
omp_init_nest_lock_with_hint_(omp_nest_lock_t *arg, omp_lock_hint_t hint)
{
  omp_init_nest_lock(arg);
}


double
omp_get_wtime(void)
{
  /* This function does not provide a working
   * wallclock timer. Replace it with a version
   * customized for the target machine.
   */
  return 0.0;
}

double
omp_get_wtime_(void)
{
  /* This function does not provide a working
   * wallclock timer. Replace it with a version
   * customized for the target machine.
   */
  return 0.0;
}

double
omp_get_wtick(void)
{
  /* This function does not provide a working
   * clock tick function. Replace it with
   * a version customized for the target machine.
   */
  return 365. * 86400.;
}

double
omp_get_wtick_(void)
{
  /* This function does not provide a working
   * clock tick function. Replace it with
   * a version customized for the target machine.
   */
  return 365. * 86400.;
}

int
omp_in_final(void)
{
  return 1;
}

kmp_int32
__kmpc_global_thread_num(void *id)
{
  return 0;
}

kmp_int32
__kmpc_global_thread_num_(void *id)
{
  return 0;
}

void
__kmpc_critical(ident_t *id, kmp_int32 tn, kmp_critical_name *sem)
{
}

void
__kmpc_critical_(ident_t *id, kmp_int32 *tn, kmp_critical_name *sem)
{
}

void
__kmpc_end_critical(ident_t *id, kmp_int32 tn, kmp_critical_name *sem)
{
}

void
__kmpc_end_critical_(ident_t *id, kmp_int32 *tn, kmp_critical_name *sem)
{
}

void *
__kmpc_threadprivate_cached(ident_t *id, kmp_int32 tn, void *data, size_t size,
                            void ***cache)
{
  return (void *)0;
}

void *
__kmpc_threadprivate_cached_(ident_t *id, kmp_int32 *tn, void *data,
                             size_t *size, void ***cache)
{
  return (void *)0;
}

void
__kmpc_barrier(ident_t *id, kmp_int32 tn)
{
}

void
__kmpc_barrier_(ident_t *id, kmp_int32 *tn)
{
}

void *
__kmpc_threadprivate(ident_t *id, kmp_int32 tn, void *data, size_t size)
{
  return (void *)0;
}

void *
__kmpc_threadprivate_(ident_t *id, kmp_int32 tn, void *data, size_t size)
{
  return (void *)0;
}

void
__kmpc_fork_call(ident_t *loc, kmp_int32 argc, void *microtask, ...)
{
}

void
__kmpc_for_static_init_8(ident_t *loc, kmp_int32 gtid, kmp_int32 schedtype, kmp_int32 *plastiter,
                      kmp_int64 *plower, kmp_int64 *pupper,
                      kmp_int64 *pstride, kmp_int64 incr, kmp_int64 chunk )
{
}

void
__kmpc_push_num_threads(ident_t *loc, kmp_int32 global_tid, kmp_int32 num_threads)
{
}
