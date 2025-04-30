/*
 * Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
 * See https://llvm.org/LICENSE.txt for license information.
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN64
#include <pthread.h>
#endif
#include "komp.h"
#include <stdioInterf.h>

/* This routine makes a simple omp library call to force lazy initialization of
 * kmpc to occur early.
 */

void
_mp_p(kmp_critical_name *sem)
{
  __kmpc_critical(0, __kmpc_global_thread_num(0), sem);
}

void
_mp_v(kmp_critical_name *sem)
{
  __kmpc_end_critical(0, __kmpc_global_thread_num(0), sem);
}

void
_mp_ptest(omp_lock_t *sem)
{
}

static kmp_critical_name sem_cs;

void
_mp_bcs(void)
{
  __kmpc_critical(0, __kmpc_global_thread_num(0), &sem_cs);
}

void
_mp_ecs(void)
{
  __kmpc_end_critical(0, __kmpc_global_thread_num(0), &sem_cs);
}

static kmp_critical_name sem_stdio;

void
_mp_bcs_stdio(void)
{
  __kmpc_critical(0, __kmpc_global_thread_num(0), &sem_stdio);
}

void
_mp_ecs_stdio(void)
{
  __kmpc_end_critical(0, __kmpc_global_thread_num(0), &sem_stdio);
}

static kmp_critical_name nest_sem;
static omp_nest_lock_t nest_lock;

static int is_init_nest = 0;
static int is_init_nest_red = 0;
static int is_atfork_registered = 0;

static void
__llcrit_atfork(void)
{
  is_init_nest = 0;
  is_init_nest_red = 0;
  /* The atfork handlers are inherited by the sub-processes,
   * see https://elias.rhi.hi.is/libc/Threads-and-Fork.html
   */
  if (!is_atfork_registered)
    fprintf(__io_stderr(), "The atfork not registered when it should be!\n");
}

void
_mp_bcs_nest(void)
{
  if (!is_init_nest) {
    _mp_p(&nest_sem);
    if (!is_init_nest) {
#ifndef _WIN64
      if (!is_atfork_registered) {
        if (pthread_atfork(NULL, NULL, __llcrit_atfork))
          fprintf(__io_stderr(), "Could not register atfork handler!\n");
        else
          is_atfork_registered = 1;
      }
#endif
      omp_init_nest_lock(&nest_lock);
      is_init_nest = 1;
    }
    _mp_v(&nest_sem);
  }
  omp_set_nest_lock(&nest_lock);
}

void
_mp_ecs_nest(void)
{
  omp_unset_nest_lock(&nest_lock);
}

// This lock is used only for the reduction, using the same locks
// `_mp_bcs/ecs` was causing deadlocks when a function that contains a
// reductions was being called directly in a print/write statement
// (those locks are used to make the print thread safe and when used
// in conjunction with a reduction the same lock was being called
// twice by different threads causing the deadlock)
static kmp_critical_name nest_sem_red;
static omp_nest_lock_t nest_lock_red;

void
_mp_bcs_nest_red(void)
{
  if (!is_init_nest_red) {
    _mp_p(&nest_sem_red);
    if (!is_init_nest_red) {
#ifndef _WIN64
      if (!is_atfork_registered) {
        if (pthread_atfork(NULL, NULL, __llcrit_atfork))
          fprintf(__io_stderr(), "Could not register atfork handler!\n");
        else
          is_atfork_registered = 1;
      }
#endif
      omp_init_nest_lock(&nest_lock_red);
      is_init_nest_red = 1;
    }
    _mp_v(&nest_sem_red);
  }
  omp_set_nest_lock(&nest_lock_red);
}

void
_mp_ecs_nest_red(void)
{
  omp_unset_nest_lock(&nest_lock_red);
}
// end reduction locks

/* allocate and initialize a thread-private common block */

void
_mp_cdeclp(void *blk, void ***blk_tp, int size)
{
  __kmpc_threadprivate_cached(0, __kmpc_global_thread_num(0), (void *)blk,
                              (size_t)size, blk_tp);
}

void
_mp_cdecli(void *blk, void ***blk_tp, int size)
{
  __kmpc_threadprivate_cached(0, __kmpc_global_thread_num(0), (void *)blk,
                              (size_t)size, blk_tp);
}

void
_mp_cdecl(void *blk, void ***blk_tp, int size)
{
  __kmpc_threadprivate_cached(0, __kmpc_global_thread_num(0), (void *)blk,
                              (size_t)size, blk_tp);
}

static char *singadr;
static long singlen;

/* C/C++: copy a private stack or other other variable */
void
_mp_copypriv(char *adr, long len, int thread)
{
  if (thread == 0) {
    singadr = adr;
    singlen = len;
  }
  __kmpc_barrier(0, __kmpc_global_thread_num(0));
  if (thread)
    memcpy(adr, singadr, singlen);
  __kmpc_barrier(0, __kmpc_global_thread_num(0));
}

/* copy allocatable data from the one thread to another */

void
_mp_copypriv_al(char **adr, long len, int thread)
{

  if (thread == 0) {
    singadr = *adr;
    singlen = len;
  }
  __kmpc_barrier(0, __kmpc_global_thread_num(0));
  if (thread)
    memcpy(*adr, singadr, singlen);
  __kmpc_barrier(0, __kmpc_global_thread_num(0));

  /* reason for second barrier is that we want to wait until every thread
   * is done copying because we have only one singadr
   * if we have another mp_copypriv... we don't want to overwrite singadr
   */
}

/* C/C++: copy data from the threads' block to the other threads blocks */

void
_mp_copypriv_move(void *blk_tp, int off, int size, int single_thread)
{
  int lcpu;
  char *to;
  char *garbage = 0;

  if (single_thread != -1) { /* single thread */
    singadr = __kmpc_threadprivate_cached(0, single_thread, garbage,
                                          (size_t)size, blk_tp);
    singlen = size;
  }
  __kmpc_barrier(0, __kmpc_global_thread_num(0));
  if (single_thread == -1) { /* single thread */
    lcpu = __kmpc_global_thread_num(0);
    to = __kmpc_threadprivate_cached(0, lcpu, garbage, (size_t)size, blk_tp);
    memcpy(to, singadr, size);
  }
  __kmpc_barrier(0, __kmpc_global_thread_num(0));
}

/* C/C++: copy data from the threads' block to the other threads blocks.
 * Use when experiment flag 69,0x80
 */

void
_mp_copypriv_move_tls(void **blk_tp, int off, int size, int single_thread)
{
  int lcpu;
  char *to;
  char *garbage = 0;

  if (single_thread != -1) { /* single thread */
    if (*blk_tp == 0)
      singadr =
          (char *)__kmpc_threadprivate(0, single_thread, garbage, (size_t)size);
    else
      singadr = *blk_tp;
    singlen = size;
  }
  __kmpc_barrier(0, __kmpc_global_thread_num(0));
  if (single_thread == -1) { /* single thread */
    lcpu = __kmpc_global_thread_num(0);
    if (*blk_tp == 0)
      to = __kmpc_threadprivate(0, lcpu, garbage, (size_t)size);
    else
      to = *blk_tp;
    memcpy(to, singadr, size);
  }
  __kmpc_barrier(0, __kmpc_global_thread_num(0));
}

/*C: copy data from the master's block to the other threads blocks
 * Don't use: keep for backward compatibility
 */

void
_mp_copyin_move(void *blk_tp, int off, int size)
{
  int lcpu;
  char *to, *fr;
  char *garbage = 0;

  lcpu = __kmpc_global_thread_num(0);

  if (lcpu != 0) {
    fr = __kmpc_threadprivate_cached(0, 0, garbage, (size_t)size, blk_tp);
    to = __kmpc_threadprivate_cached(0, lcpu, garbage, (size_t)size, blk_tp);
    if (to != fr)
      memcpy(to, fr, size);
  }
  __kmpc_barrier(0, __kmpc_global_thread_num(0));
}

/* C: copy data from the master's block to the other threads blocks
 * Use when experiment flag 69,0x80
 */

void
_mp_copyin_move_tls(void *blk_tp, int off, int size)
{
  int lcpu;
  char *to, *fr;
  char *garbage = 0;

  lcpu = __kmpc_global_thread_num(0);

  if (lcpu != 0) {
    fr = __kmpc_threadprivate(0, 0, garbage, (size_t)size);
    to = __kmpc_threadprivate(0, lcpu, garbage, (size_t)size);
    if (to != fr)
      memcpy(to, fr, size);
  }
  __kmpc_barrier(0, __kmpc_global_thread_num(0));
}

typedef void (*assign_func_ptr)(void *, void *);

/* C++: copy data from the master's block to the other threads blocks
   using the assignment operator
   vector_size is 1 for non arrays
                  n for array[n]
 * Don't use: keep for backward compatibility
 */

void
_mp_copyin_move_cpp(void *blk_tp, int off, int class_size, int vector_size,
                    assign_func_ptr assign_op)
{
  int lcpu;
  char *to, *fr;
  char *garbage = 0;
  int i;

  lcpu = __kmpc_global_thread_num(0);

  __kmpc_barrier(0, lcpu);
  if (lcpu != 0) {
    fr = __kmpc_threadprivate_cached(
        0, 0, garbage, (size_t)(class_size * vector_size), blk_tp);
    to = __kmpc_threadprivate_cached(
        0, lcpu, garbage, (size_t)(class_size * vector_size), blk_tp);

    for (i = 0; i < vector_size; i++) {
      if (to != fr)
        (*assign_op)(to, fr);
      to += class_size;
      fr += class_size;
    }
  }
  __kmpc_barrier(0, lcpu);
}

/* C++: copy data from the master's block to the other threads blocks
   using the assignment operator
        vector_size is 1 for non arrays
                       n for array[n]
 */
void
_mp_copyin_move_cpp_new(void *blk_tp, int off, int class_size, int vector_size,
                        assign_func_ptr assign_op, char *fr)
{
  int lcpu;
  char *to;
  char *garbage = 0;
  int i;

  if (!fr)
    return;

  lcpu = __kmpc_global_thread_num(0);

  to = __kmpc_threadprivate_cached(0, lcpu, garbage,
                                   (size_t)(class_size * vector_size), blk_tp);

  for (i = 0; i < vector_size; i++) {
    if (to != fr)
      (*assign_op)(to, fr);
    to += class_size;
    fr += class_size;
  }
}

/*
 * Use when experiment flag 69,0x80
 */
void
_mp_copyin_move_cpp_tls(void *master, void *slave, int class_size,
                        int vector_size, assign_func_ptr assign_op)
{
  char *to, *fr;
  int i;

  fr = (char *)master;
  to = (char *)slave;
  if (fr && to) {
    for (i = 0; i < vector_size; i++) {
      (*assign_op)(to, fr);
      to += class_size;
      fr += class_size;
    }
  }
}

/* Copy multiple items from master to children threads.
 * Don't use: keep for backward compatibility
 */
void
_mp_copyin_move_multiple(int n_entries, void *data)
{
  int i;
  const int tid = __kmpc_global_thread_num(NULL);
  struct pair_t {
    size_t size;
    void *data;
  };

  if (tid != 0) {
    for (i = 0; i < n_entries; ++i) {
      struct pair_t *item = (struct pair_t *)data + i;
      void *key = item->data;
      const size_t size = item->size;
      void *to = __kmpc_threadprivate_cached(NULL, tid, NULL, size, key);
      /* FIXME: Should this be 0 or the team master?
       * I think the gtid of team master.
       */
      void *fr = __kmpc_threadprivate_cached(NULL, 0, NULL, size, key);
      if (to != fr)
        memcpy(to, fr, size);
    }
  }

  __kmpc_barrier(0, tid);
}

/* copy allocatable data from the master's block to the other threads' blocks */

void
_mp_copyin_move_al(void *blk_tp, int off, long size)
{
  int lcpu;
  char *to, *fr;
  char *garbage = 0;

  lcpu = __kmpc_global_thread_num(0);
  if (lcpu != 0) {
    fr = __kmpc_threadprivate_cached(0, 0, garbage, (size_t)size, blk_tp);
    to = __kmpc_threadprivate_cached(0, lcpu, garbage, (size_t)size, blk_tp);
    if (to && to != fr) {
      memcpy(to, fr, size);
    }
  }
  __kmpc_barrier(0, __kmpc_global_thread_num(0));
}

/* Handler for __kmpc_copyprivate 'cpy_func'
 * See how we marshall data in make_copypriv_array()  in expsmp.c.
 */
void
_mp_copypriv_kmpc(void *dest, void *src)
{
  struct pair_t {
    size_t *size;
    void *data;
  };
  const struct pair_t *to = (struct pair_t *)dest;
  const struct pair_t *from = (struct pair_t *)src;

  for (; from->size; ++from, ++to) {
    if (to->data != from->data)
      memcpy(to->data, from->data, *from->size);
  }
}

/* duplicate kmpc_threadprivate_cached but we assume each thread has its own
 * addr in its own [tls] address space so that it does not need to access memory
 * in other thread's area. Use when experiment flag 69,0x80
 */
void *
_mp_get_threadprivate(ident_t *ident, kmp_int32 gtid, void *tpv, size_t size,
                      void **addr)
{
  if (*addr == NULL) {
    *addr = __kmpc_threadprivate(ident, gtid, tpv, size);
  }
  return *addr;
}
