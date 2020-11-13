
#ifndef BOLT_SCHEDULING_UTIL_H
#define BOLT_SCHEDULING_UTIL_H

#include "omp_testsuite.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sched.h>

void check_num_ess(int desired) {
  int num_xstreams;
  ABT_EXIT_IF_FAIL(ABT_xstream_get_num(&num_xstreams));
  if (num_xstreams != desired) {
    printf("check_num_ess: num_xstreams (%d) != desired (%d)\n", num_xstreams,
           desired);
    exit(1);
  }
}

typedef struct {
  int counter, flag;
} timeout_barrier_t;

void timeout_barrier_init(timeout_barrier_t *barrier) {
  barrier->counter = 0;
  barrier->flag = 0;
}

void timeout_barrier_wait(timeout_barrier_t *barrier, int num_waiters) {
  const int timeout_ms = 4000;
  const int wait_ms = 200;

  // return 1 if failed.
  int *p_counter = &barrier->counter;
  int *p_flag = &barrier->flag;

  if (__atomic_add_fetch(p_counter, 1, __ATOMIC_ACQ_REL) == num_waiters) {
    double start_time = omp_get_wtime();
    while (omp_get_wtime() < start_time + wait_ms / 1000.0) {
      if (__atomic_load_n(p_counter, __ATOMIC_ACQUIRE) != num_waiters) {
        printf("timeout_barrier_wait: # of arrivals > num_waiters (%d)\n", num_waiters);
        exit(1);
      }
      sched_yield();
    }
    // Going back to the normal barrier implementation.
    __atomic_store_n(p_flag, 1, __ATOMIC_RELEASE);
    // wait until current_counter gets 1
    do {
      // This does not require timeout.
      sched_yield();
    } while (__atomic_load_n(p_counter, __ATOMIC_ACQUIRE) != 1);
    // update a flag again.
    __atomic_store_n(p_counter, 0, __ATOMIC_RELEASE);
    __atomic_store_n(p_flag, 0, __ATOMIC_RELEASE);
  } else {
    double start_time = omp_get_wtime();
    do {
      if (omp_get_wtime() > start_time + (timeout_ms + wait_ms) / 1000.0) {
        printf("timeout_barrier_wait: timeout expires (%d)\n",
               (int)__atomic_load_n(p_counter, __ATOMIC_ACQUIRE));
        exit(1);
      }
      sched_yield();
    } while (__atomic_load_n(p_flag, __ATOMIC_ACQUIRE) == 0);
    // now p_flag is 1. Let's decrease the counter.
    __atomic_sub_fetch(p_counter, 1, __ATOMIC_ACQ_REL);
    // wait until p_flag gets 0.
    do {
      // This does not require timeout.
      sched_yield();
    } while (__atomic_load_n(p_flag, __ATOMIC_ACQUIRE) == 1);
  }
}

#endif // BOLT_SCHEDULING_UTIL_H
