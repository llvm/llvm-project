/*
 * kmp_abt.h -- header file.
 */

//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.txt for details.
//
//===----------------------------------------------------------------------===//

#ifndef KMP_ABT_H
#define KMP_ABT_H

#if KMP_USE_ABT

#include <abt.h>

#define KMP_ABT_FORK_NUM_WAYS_DEFAULT 2
#define KMP_ABT_FORK_CUTOFF_DEFAULT (1 << 20)
#define KMP_ABT_SCHED_SLEEP_DEFAULT 0
#define KMP_ABT_SCHED_MIN_SLEEP_NSEC_DEFAULT 1 /* 1ns */
#define KMP_ABT_SCHED_MAX_SLEEP_NSEC_DEFAULT 1048576 /* 1ms */
#define KMP_ABT_SCHED_EVENT_FREQ_DEFAULT 256 /* Every 100 work-stealing loops */
#define KMP_ABT_SCHED_EVENT_FREQ_MAX 1048576 /* Needed to avoid deadlock */
#define KMP_ABT_WORK_STEAL_FREQ_DEFAULT 65536 /* Every 65536 loops */

static inline uint32_t __kmp_abt_fast_rand32(uint32_t *p_seed) {
  // George Marsaglia, "Xorshift RNGs", Journal of Statistical Software,
  // Articles, 2003
  uint32_t seed = *p_seed;
  seed ^= seed << 13;
  seed ^= seed >> 17;
  seed ^= seed << 5;
  *p_seed = seed;
  return seed;
}

// ES-local data.
typedef struct kmp_abt_local {
  /* ------------------------------------------------------------------------ */
  // Mostly read only

  ABT_xstream xstream;
  ABT_sched sched;

  // Scheduler
  ABT_pool shared_pool;
  int place_id;
  ABT_pool place_pool;
  /* ------------------------------------------------------------------------ */
} __attribute__((aligned(CACHE_LINE))) kmp_abt_local_t;

// Global data.
typedef struct kmp_abt_global {
  /* ------------------------------------------------------------------------ */
  // Mostly read only
  int num_xstreams;
  int fork_num_ways;
  int fork_cutoff;
  int is_sched_sleep;
  int sched_sleep_min_nsec;
  int sched_sleep_max_nsec;
  int sched_event_freq;
  uint32_t work_steal_freq;
  int num_places;
  ABT_pool *place_pools;

  // ES-local data.
  kmp_abt_local *locals;
} __attribute__((aligned(CACHE_LINE))) kmp_abt_global_t;

extern kmp_abt_global_t __kmp_abt_global;

typedef struct kmp_abt_affinity_place {
  size_t num_ranks;
  int *ranks;
} kmp_abt_affinity_place_t;

typedef struct kmp_abt_affinity_places {
  size_t num_places;
  kmp_abt_affinity_place_t **p_places;
} kmp_abt_affinity_places_t;

extern kmp_abt_affinity_places_t *__kmp_abt_parse_affinity(int num_xstreams,
                                                           const char *str,
                                                           size_t len,
                                                           bool verbose);
extern int __kmp_abt_affinity_place_find
    (const kmp_abt_affinity_place_t *p_place, int rank);
extern void __kmp_abt_affinity_places_free(kmp_abt_affinity_places_t *p_places);

#endif // KMP_USE_ABT
#endif // KMP_ABT_H
