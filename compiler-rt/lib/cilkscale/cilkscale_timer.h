#ifndef INCLUDED_CILKSCALE_TIMER_H
#define INCLUDED_CILKSCALE_TIMER_H

#define RDTSC 1
#define CLOCK 2
// TODO: To use the INST timer with the -fcilktool=cilkscale flag, update the
// getCSIOptionsForCilkscale() function in
// tools/clang/lib/CodeGen/BackendUtil.cpp to re-enable CSI instrumentation of
// basic blocks.  This instrumentation is disabled by default, because it has a
// significant impact on the performance of Cilkscale.
#define INST 3

#ifndef CSCALETIMER
// Valid cilkscale timer values are RDTSC, CLOCK, and INST
#define CSCALETIMER CLOCK
#endif

#if CSCALETIMER == RDTSC
#include <x86intrin.h>
#elif CSCALETIMER == CLOCK
#include <time.h>
/* #define _POSIX_C_SOURCE 200112L */
#endif

/**
 * Data structures and helper methods for time of user strands.
 */

#if CSCALETIMER == RDTSC
typedef uint64_t cilkscale_timer_t;
#elif CSCALETIMER == CLOCK
typedef struct timespec cilkscale_timer_t;
#else
typedef uint64_t cilkscale_timer_t;
#endif

static inline uint64_t elapsed_nsec(const cilkscale_timer_t *stop,
                                    const cilkscale_timer_t *start) {
#if CSCALETIMER == RDTSC
  return *stop - *start;
#elif CSCALETIMER == CLOCK
  return (uint64_t)(stop->tv_sec - start->tv_sec) * 1000000000ll
    + (stop->tv_nsec - start->tv_nsec);
#else
  return 0;
#endif
}

static inline void gettime(cilkscale_timer_t *timer) {
#if CSCALETIMER == RDTSC
  *timer = __rdtsc();
#elif CSCALETIMER == CLOCK
  // TB 2014-08-01: This is the "clock_gettime" variant I could get
  // working with -std=c11.  I want to use TIME_MONOTONIC instead, but
  // it does not appear to be supported on my system.
  /* timespec_get(timer, TIME_UTC); */
  clock_gettime(CLOCK_MONOTONIC, timer);
#endif
}

static inline void settime(cilkscale_timer_t *timer, cilkscale_timer_t val) {
  *timer = val;
}

static inline void printtime(uint64_t timeval) {
#if CSCALETIMER == RDTSC
  fprintf(stderr, "%f Gcycles", timeval/(1000000000.0));
#elif CSCALETIMER == CLOCK
  fprintf(stderr, "%fs", timeval/(1000000000.0));
#elif CSCALETIMER == INST
  fprintf(stderr, "%f Minstructions", timeval/(1000000.0));
  /* fprintf(stderr, "%ld inst", timeval); */
#endif
}

static inline void get_bb_time(uint64_t *running_wrk, uint64_t *contin_spn,
                               const csi_id_t bb_id) {
#if CSCALETIMER == INST
  uint64_t inst_count = __csi_get_bb_sizeinfo(bb_id)->non_empty_size;
  *running_wrk += inst_count;
  *contin_spn += inst_count;
#endif
}  

#endif // INCLUDED_CILKSCALE_TIMER_H
