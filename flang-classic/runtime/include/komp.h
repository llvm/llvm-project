/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#ifndef _PGOMP_H
#define _PGOMP_H

#include <stdint.h>

/* simple lock */

/* lock API functions */
typedef struct omp_lock_t {
  void *_lk;
} omp_lock_t;

/* KMPC nested lock API functions */
typedef struct omp_nest_lock_t {
  void *_lk;
} omp_nest_lock_t;

typedef enum omp_sched_t {
  omp_sched_static = 1,
  omp_sched_dynamic = 2,
  omp_sched_guided = 3,
  omp_sched_auto = 4
} omp_sched_t;

/* OpenMP 4.0 affinity API */
typedef enum omp_proc_bind_t {
  omp_proc_bind_false = 0,
  omp_proc_bind_true = 1,
  omp_proc_bind_master = 2,
  omp_proc_bind_close = 3,
  omp_proc_bind_spread = 4
} omp_proc_bind_t;

/* lock hint type for dynamic user lock */
typedef enum omp_lock_hint_t {
  omp_lock_hint_none           = 0,
  omp_lock_hint_uncontended    = 1,
  omp_lock_hint_contended      = (1<<1 ),
  omp_lock_hint_nonspeculative = (1<<2 ),
  omp_lock_hint_speculative    = (1<<3 ),
  kmp_lock_hint_hle            = (1<<16),
  kmp_lock_hint_rtm            = (1<<17),
  kmp_lock_hint_adaptive       = (1<<18)
} omp_lock_hint_t;

#ifdef __cplusplus
extern "C" {
#endif

extern void omp_set_num_threads(int n);
extern int omp_get_thread_num(void);
extern int omp_get_num_procs(void);
extern int omp_get_num_threads(void);
extern int omp_get_max_threads(void);
extern int omp_in_parallel(void);
extern int omp_in_final(void);
extern void omp_set_dynamic(int n);
extern int omp_get_dynamic(void);
extern void omp_set_nested(int n);
extern int omp_get_nested(void);
extern void omp_init_lock(omp_lock_t *s);
extern void omp_destroy_lock(omp_lock_t *s);
extern void omp_set_lock(omp_lock_t *s);
extern void omp_unset_lock(omp_lock_t *s);
extern int omp_test_lock(omp_lock_t *s);
extern void omp_init_nest_lock(omp_nest_lock_t *s);
extern void omp_destroy_nest_lock(omp_nest_lock_t *s);
extern void omp_set_nest_lock(omp_nest_lock_t *s);
extern void omp_unset_nest_lock(omp_nest_lock_t *s);
extern int omp_test_nest_lock(omp_nest_lock_t *s);
extern double omp_get_wtime(void);
extern double omp_get_wtick(void);
extern long omp_get_stack_size(void);
extern void omp_set_stack_size(long l);
extern int omp_get_thread_limit(void);
extern void omp_set_max_active_levels(int);
extern int omp_get_max_active_levels(void);
extern int omp_get_level(void);
extern int omp_get_ancestor_thread_num(int);
extern int omp_get_team_size(int);
extern int omp_get_active_level(void);
extern void omp_set_schedule(omp_sched_t, int);
extern void omp_get_schedule(omp_sched_t *, int *);

typedef int _Atomic_word;
extern void _mp_atomic_add(int *, int);
extern void _mp_exchange_and_add(int *, int);

/* hinted lock initializers */
extern void omp_init_lock_with_hint(omp_lock_t *, omp_lock_hint_t);
extern void omp_init_nest_lock_with_hint(omp_nest_lock_t *, omp_lock_hint_t);

/* OpenMP 4.0 */
extern int  omp_get_default_device (void);
extern void omp_set_default_device (int);
extern int  omp_is_initial_device (void);
extern int  omp_get_num_devices (void);
extern int  omp_get_num_teams (void);
extern int  omp_get_team_num (void);
extern int  omp_get_cancellation (void);
extern omp_proc_bind_t omp_get_proc_bind (void);
extern int omp_get_place_num_procs(int place_num);
extern void omp_get_place_proc_ids(int place_num, int *ids);
extern int omp_get_place_num(void);
extern int omp_get_partition_num_places(void);
extern void omp_get_partition_place_nums(int *place_nums);
extern void omp_set_default_device(int device_num);
extern int omp_get_default_device(void);
extern int omp_get_num_devices(void);
extern int omp_get_num_teams(void);
extern int omp_get_team_num(void);
extern int omp_is_initial_device(void);
extern int omp_get_initial_device(void);
extern int omp_get_max_task_priority(void);
extern void omp_init_lock_with_hint(omp_lock_t *lock, omp_lock_hint_t hint);
extern void omp_init_nest_lock_with_hint(omp_nest_lock_t *lock, omp_lock_hint_t hint);
extern void *omp_target_alloc(size_t size, int device_num);
extern void omp_target_free(void * device_ptr, int device_num);
extern int omp_target_is_present(void * ptr, int device_num);
extern int omp_target_memcpy(
  void *dst, 
  void *src, 
  size_t length, 
  size_t dst_offset, 
  size_t src_offset, 
  int dst_device_num, 
  int src_device_num);

extern int omp_target_memcpy_rect( 
 void *dst, 
 void *src, 
 size_t element_size, 
 int num_dims,
 const size_t *volume,
 const size_t *dst_offsets,
 const size_t *src_offsets,
 const size_t *dst_dimensions,
 const size_t *src_dimensions,
 int dst_device_num, int src_device_num);

extern int omp_target_associate_ptr(
 void * host_ptr, 
 void * device_ptr,
 size_t size,
 size_t device_offset,
 int device_num);

extern int omp_target_disassociate_ptr(void * ptr, int device_num);


#ifdef __cplusplus
/* used to call omp_init_lock ( ) as a static initializer
   when protecting locally scoped statics */
class __cplus_omp_init_lock_t
{

  omp_lock_t omp_lock;

public:
  __cplus_omp_init_lock_t() { omp_init_lock(&omp_lock); }
};
}
#endif

/*
 * ident_t; kmpc's ident structure that describes a source location; see
 *    kmp.h "typedef struct ident"
 */
typedef void ident_t;

#ifndef __WORDSIZE
#ifdef _WIN32
#ifdef _WIN64
#define __WORDSIZE 64
#else
#define __WORDSIZE 32
#endif
#else
#error "Unknown word size!"
#endif
#endif

typedef int kmp_int32;
typedef long long kmp_int64;
#if __WORDSIZE == 32
typedef kmp_int32 kmp_critical_name[8];  /* must be 32 bytes */
#else
typedef kmp_int64 kmp_critical_name[4];  /* must be 32 bytes and 8-byte aligned */
#endif

extern kmp_int32 __kmpc_global_thread_num(ident_t *);
extern void __kmpc_critical(ident_t *, kmp_int32, kmp_critical_name *);
extern void __kmpc_end_critical(ident_t *, kmp_int32, kmp_critical_name *);
extern void* __kmpc_threadprivate_cached(ident_t *, kmp_int32, void*, size_t, void*** );
extern void* __kmpc_threadprivate(ident_t *, kmp_int32, void*, size_t);
extern void __kmpc_barrier(ident_t *, kmp_int32);

#endif /*_PGOMP_H*/
