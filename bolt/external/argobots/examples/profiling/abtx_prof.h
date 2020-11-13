/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * How to use ABTX_profiler
 *
 * This header file implements an example of a simple profiler that measures the
 * basic performance numbers.  This example profiler should be sufficiently
 * useful in practice.
 *
 *  1. Compile Argobots with --enable-tool
 *  2. Modify your program as follows:
 *   2.1 Copy this header file and include it.
 *   2.2 Right after ABT_init(), call ABTX_prof_init().
 *   2.3 When you want to start a profiler, call ABTX_prof_start().
 *   2.4 When you want to stop a profiler, call ABTX_prof_stop().
 *   2.5 Call ABTX_prof_print() to print out the obtained results.
 *   2.6 Repeat 2.3 - 2.5 if needed.
 *   2.7 Right before ABT_finalize(), call ABTX_prof_finalize().
 *  3. Compile your program and run it with tool-enabled Argobots.  You might
 *     need to add -lpthread to compile your code with this header.
 *
 * Your program will be modified as follows:
 *
 * ============================================================================
 *
 * #include "abtx_prof.h"                                  // Added
 *
 * ABTX_prof_context g_prof_context;                       // Added
 * int main_func() {
 *   ...
 *   ABT_init(args, argc);
 *   ABTX_prof_init(&g_prof_context)                       // Added
 *   ...
 *   ABTX_prof_start(g_prof_context, ABTX_PROF_MODE_BASIC) // Added
 *   for (int iter = 0; iter < niters; iter++) {
 *     ...
 *     ...
 *     ...
 *   }
 *   ABTX_prof_stop(g_prof_context);                       // Added
 *   ABTX_prof_print(g_prof_context, stdout,
 *                   ABTX_PRINT_MODE_SUMMARY
 *                   | ABTX_PRINT_MODE_FANCY);             // Added
 *   ...
 *   ABTX_prof_finalize(g_prof_context)                    // Added
 *   ABT_finalize();
 *   ...
 * }
 *
 * ============================================================================
 *
 * - Profiling mode (prof_mode)
 *
 *   Users can pass either ABTX_PROF_MODE_BASIC or ABTX_PROF_MODE_DETAILED.
 *   ABTX_PROF_MODE_BASIC is lighter but can gather less information.
 *   ABTX_PROF_MODE_DETAILED is quite heavy, so please use ABTX_PROF_MODE_BASIC
 *   if it is enough (and in many cases, it should be enough).
 *
 * - Print mode (print_mode)
 *
 *   Users can pass [either ABTX_PRINT_MODE_RAW or ABTX_PRINT_MODE_SUMMARY] and
 *   [either ABTX_PRINT_MODE_CSV or ABTX_PRINT_MODE_FANCY].  For the first
 *   trial, "(ABTX_PRINT_MODE_SUMMARY | ABTX_PRINT_MODE_FANCY)" (where "|" is a
 *   bitwise OR) is recommended.
 *
 *   ABTX_PRINT_MODE_SUMMARY prints the common performance metrics, while
 *   ABTX_PRINT_MODE_RAW shows the raw performance data.  If users do not deeply
 *   understand the mechanism of Argobots, ABTX_PRINT_MODE_SUMMARY is
 *   recommended.
 *
 *   ABTX_PRINT_MODE_CSV outputs the data in a CSV format while the text
 *   displayed by ABTX_PRINT_MODE_FANCY is easier to read in the terminal.
 *
 * Note that profiling imposes certain overheads.  For accurate performance
 * analysis, please compare the performance with the original performance
 * without a profiler.
 */

#ifndef ABTX_PROF_H_INCLUDED
#define ABTX_PROF_H_INCLUDED

#include <stdio.h>

#define ABTX_PROF_MODE_BASIC 0
#define ABTX_PROF_MODE_DETAILED 1

#define ABTX_PRINT_MODE_RAW 0x1
#define ABTX_PRINT_MODE_SUMMARY 0x2
#define ABTX_PRINT_MODE_CSV 0x4
#define ABTX_PRINT_MODE_FANCY 0x8

typedef struct ABTX_prof_context_opaque *ABTX_prof_context;

static int ABTX_prof_init(ABTX_prof_context *p_new_context);
static int ABTX_prof_start(ABTX_prof_context context, int prof_mode);
static int ABTX_prof_stop(ABTX_prof_context context);
static int ABTX_prof_clean(ABTX_prof_context context);
static int ABTX_prof_print(ABTX_prof_context context, FILE *stream,
                           int print_mode);
static int ABTX_prof_finalize(ABTX_prof_context context);

/*
 * Profiler configurations that might affect compilation.
 *
 * - ABTX_PROF_USE_BUILTIN_EXPECT
 *   Set 0 if your compiler does not support __builtin_expect().  This builtin
 *   should be supported by sufficiently new GCC, Clang, ICC, and XLC compilers.
 *   Setting it to 0 may result in poor code optimization.
 *
 * - ABTX_PROF_USE_ALWAYS_INLINE
 *   Set 0 if your compiler does not support __attribute__((always_inline)).
 *   This attribute should be supported by sufficiently new GCC, Clang, ICC, and
 *   XLC compilers.
 *
 * - ABTX_PROF_ASSUME_SCHED_ALWAYS_ACTIVE
 *   Set 0 if schedulers might be not scheduled in your program because of
 *   oversubscription of OS-level threads (=Pthreads) or scheduler sleep.
 *   Although many applications create as many execution streams as a number of
 *   cores and let them burn all the CPU resources, this assumption does not
 *   hold in some cases.  for example, if Argobots is used as a backend event
 *   engine, Argobots should avoid wasting CPU cores if there is no work.
 *   Setting 0 enables the profiler to use a heavy per-thread timer.
 *
 * - ABTX_PROF_USE_HARDWARE_CYCLES
 *   Set 0 if your compiler does not support assembly code that gets
 *   architecture-dependent hardware clock.  This code should work with
 *   sufficiently new GCC, Clang, ICC, and XLC compilers.
 *
 */

#ifndef ABTX_PROF_USE_BUILTIN_EXPECT
#define ABTX_PROF_USE_BUILTIN_EXPECT 1
#endif

#ifndef ABTX_PROF_USE_ALWAYS_INLINE
#define ABTX_PROF_USE_ALWAYS_INLINE 1
#endif

#ifndef ABTX_PROF_ASSUME_SCHED_ALWAYS_ACTIVE
#define ABTX_PROF_ASSUME_SCHED_ALWAYS_ACTIVE 1
#endif

#ifndef ABTX_PROF_USE_HARDWARE_CYCLES
#define ABTX_PROF_USE_HARDWARE_CYCLES 1
#endif

/*
 * Internal implementation.  This should not be modified by profiler users.
 */

#include <stdint.h>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <assert.h>
#include <stdarg.h>
#include <abt.h>

#if ABTX_PROF_USE_BUILTIN_EXPECT
#define ABTXI_prof_likely(cond) __builtin_expect(!!(cond), 1)
#define ABTXI_prof_unlikely(cond) __builtin_expect(!!(cond), 0)
#else
#define ABTXI_prof_likely(cond) (cond)
#define ABTXI_prof_unlikely(cond) (cond)
#endif

#if ABTX_PROF_USE_ALWAYS_INLINE
#define ABTXI_prof_always_inline inline __attribute__((always_inline))
#else
#define ABTXI_prof_always_inline inline
#endif

#undef ABTXI_PROF_USE_SYNC_BUILTIN
#if defined(__PGIC__) || defined(__ibmxl__)
/* Their __atomic implementations are not trustworthy.  See #162 and #211. */
#define ABTXI_PROF_USE_SYNC_BUILTIN
#endif

#ifndef ABTXI_PROF_MEM_BLOCK_SIZE
#define ABTXI_PROF_MEM_BLOCK_SIZE (32 * 1024) /* bytes */
#endif
#ifndef ABTXI_PROF_MAX_DEPTH
#define ABTXI_PROF_MAX_DEPTH 4
#endif
#ifndef ABTXI_PROF_DEFAULT_NUM_XSTREAMS
#define ABTXI_PROF_DEFAULT_NUM_XSTREAMS 32
#endif

#undef ABTXI_PROF_USE_CYCLES

#if ABTX_PROF_USE_HARDWARE_CYCLES
/* Use "faster" hardware cycles. */
#if defined(__x86_64__)

/* x86/64 (Intel and AMD) */
static inline uint64_t ABTXI_prof_get_cycles()
{
    unsigned hi, lo;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi)::"rcx");
    uint64_t cycles = ((uint64_t)lo) | (((int64_t)hi) << 32);
    return cycles;
}
#define ABTXI_PROF_USE_CYCLES 1

#elif defined(__aarch64__)

/* 64-bit ARM */
static inline uint64_t ABTXI_prof_get_cycles()
{
    register uint64_t cycles;
    __asm__ __volatile__("isb; mrs %0, cntvct_el0" : "=r"(cycles));
    return cycles;
}
#define ABTXI_PROF_USE_CYCLES 1

#elif defined(__powerpc__)

/* POWER */
static inline uint64_t ABTXI_prof_get_cycles()
{
    register uint64_t cycles;
    __asm__ __volatile__("mfspr %0, 268" : "=r"(cycles));
    return cycles;
}
#define ABTXI_PROF_USE_CYCLES 1
/* Unknown hardware. */
#endif
#endif /* ABTX_PROF_USE_HARDWARE_CYCLES */

#ifdef ABTXI_PROF_USE_CYCLES

#define ABTXI_PROF_T int64_t
#define ABTXI_PROF_T_INVALID 0xFFFFFFFFFFFFFFFF
#define ABTXI_PROF_T_ZERO ((int64_t)0)
#define ABTXI_prof_get_time() ABTXI_prof_get_cycles()
#define ABTXI_PROF_T_STRING "HW cycles"
static double ABTXI_prof_get_time_to_sec()
{
    double t_sec1 = ABT_get_wtime();
    ABTXI_PROF_T t_start = ABTXI_prof_get_cycles();
    double t_sec2 = ABT_get_wtime();
    double t_start_s = (t_sec2 + t_sec1) / 2.0;
    while (ABT_get_wtime() < t_start_s + 1.0)
        ;
    double t_sec3 = ABT_get_wtime();
    ABTXI_PROF_T t_end = ABTXI_prof_get_cycles();
    double t_sec4 = ABT_get_wtime();
    double t_end_s = (t_sec4 + t_sec3) / 2.0;
    return (t_end_s - t_start_s) / (t_end - t_start);
}

#else

#define ABTXI_PROF_T double
#define ABTXI_PROF_T_INVALID ((double)-1.0)
#define ABTXI_PROF_T_ZERO ((double)0.0)
#define ABTXI_prof_get_time() ABT_get_wtime()
#define ABTXI_PROF_T_STRING "s"
#define ABTXI_prof_get_time_to_sec() 1.0

#endif

#if ABTX_PROF_ASSUME_SCHED_ALWAYS_ACTIVE

#define ABTXI_PROF_USE_TIME_LOCAL 0

#else

#define ABTXI_PROF_USE_TIME_LOCAL 1
#define ABTXI_PROF_LOCAL_T_INVALID ((double)-1.0)
#define ABTXI_PROF_LOCAL_T_ZERO ((double)0.0)
#define ABTXI_PROF_LOCAL_T double
static ABTXI_PROF_LOCAL_T ABTXI_prof_get_time_local()
{
    /* Return a per-thread timer. */
    struct timespec t;
    clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t);
    return t.tv_sec + t.tv_nsec * 1.0e-9;
}
#define ABTXI_PROF_LOCAL_T_STRING "s"
#define ABTXI_prof_get_local_time_to_sec() 1.0

#endif

#define ABTXI_PROF_EVENT_THREAD_CREATE 0
#define ABTXI_PROF_EVENT_THREAD_JOIN 1
#define ABTXI_PROF_EVENT_THREAD_FREE 2
#define ABTXI_PROF_EVENT_THREAD_REVIVE 3
#define ABTXI_PROF_EVENT_THREAD_RUN 4
#define ABTXI_PROF_EVENT_THREAD_FINISH 5
#define ABTXI_PROF_EVENT_THREAD_CANCEL 6
#define ABTXI_PROF_EVENT_THREAD_YIELD 7
#define ABTXI_PROF_EVENT_THREAD_SUSPEND 8
#define ABTXI_PROF_EVENT_THREAD_RESUME 9
#define ABTXI_PROF_EVENT_TASK_CREATE 10
#define ABTXI_PROF_EVENT_TASK_JOIN 11
#define ABTXI_PROF_EVENT_TASK_FREE 12
#define ABTXI_PROF_EVENT_TASK_REVIVE 13
#define ABTXI_PROF_EVENT_TASK_RUN 14
#define ABTXI_PROF_EVENT_TASK_FINISH 15
#define ABTXI_PROF_EVENT_TASK_CANCEL 16
#define ABTXI_PROF_EVENT_END_ 17

#define ABTXI_PROF_WU_TIME_THREAD_ELAPSED 0
#define ABTXI_PROF_WU_TIME_THREAD_CREATE_FIRST_RUN 1
#define ABTXI_PROF_WU_TIME_THREAD_FIRST_RUN_LAST_FINISH 2
#define ABTXI_PROF_WU_TIME_THREAD_CREATE_LAST_FINISH 3
#define ABTXI_PROF_WU_TIME_THREAD_CREATE_FREE 4
#define ABTXI_PROF_WU_TIME_TASK_ELAPSED 5
#define ABTXI_PROF_WU_TIME_TASK_CREATE_FIRST_RUN 6
#define ABTXI_PROF_WU_TIME_TASK_FIRST_RUN_LAST_FINISH 7
#define ABTXI_PROF_WU_TIME_TASK_CREATE_LAST_FINISH 8
#define ABTXI_PROF_WU_TIME_TASK_CREATE_FREE 9
#define ABTXI_PROF_WU_TIME_END_ 10

#define ABTXI_PROF_WU_LOCAL_TIME_THREAD_ELAPSED 0
#define ABTXI_PROF_WU_LOCAL_TIME_TASK_ELAPSED 1
#define ABTXI_PROF_WU_LOCAL_TIME_END_ 2

#define ABTXI_PROF_WU_COUNT_THREAD_NUM_REVIVALS 0
#define ABTXI_PROF_WU_COUNT_THREAD_NUM_YIELDS 1
#define ABTXI_PROF_WU_COUNT_THREAD_NUM_SUSPENSIONS 2
#define ABTXI_PROF_WU_COUNT_THREAD_NUM_XSTREAM_CHANGES 3
#define ABTXI_PROF_WU_COUNT_TASK_NUM_REVIVALS 4
#define ABTXI_PROF_WU_COUNT_END_ 5

#define ABTXI_PROF_TIME_LAST_RUN_INVALID ABTXI_PROF_T_INVALID
#define ABTXI_PROF_TIME_LAST_RUN_LOCAL_INVALID ABTXI_PROF_LOCAL_T_INVALID

static const char *ABTXI_get_prof_event_name(int event)
{
    switch (event) {
        case ABTXI_PROF_EVENT_THREAD_CREATE:
            return "ULT/create";
        case ABTXI_PROF_EVENT_THREAD_JOIN:
            return "ULT/join";
        case ABTXI_PROF_EVENT_THREAD_FREE:
            return "ULT/free";
        case ABTXI_PROF_EVENT_THREAD_REVIVE:
            return "ULT/revive";
        case ABTXI_PROF_EVENT_THREAD_RUN:
            return "ULT/run";
        case ABTXI_PROF_EVENT_THREAD_FINISH:
            return "ULT/finish";
        case ABTXI_PROF_EVENT_THREAD_CANCEL:
            return "ULT/cancel";
        case ABTXI_PROF_EVENT_THREAD_YIELD:
            return "ULT/yield";
        case ABTXI_PROF_EVENT_THREAD_SUSPEND:
            return "ULT/suspend";
        case ABTXI_PROF_EVENT_THREAD_RESUME:
            return "ULT/resume";
        case ABTXI_PROF_EVENT_TASK_CREATE:
            return "tasklet/create";
        case ABTXI_PROF_EVENT_TASK_JOIN:
            return "tasklet/join";
        case ABTXI_PROF_EVENT_TASK_FREE:
            return "tasklet/free";
        case ABTXI_PROF_EVENT_TASK_REVIVE:
            return "tasklet/revive";
        case ABTXI_PROF_EVENT_TASK_RUN:
            return "tasklet/run";
        case ABTXI_PROF_EVENT_TASK_FINISH:
            return "tasklet/finish";
        case ABTXI_PROF_EVENT_TASK_CANCEL:
            return "tasklet/cancel";
        default:
            return "ERR";
    }
}

static const char *ABTXI_get_prof_wu_time_name(int wu_time)
{
    switch (wu_time) {
        case ABTXI_PROF_WU_TIME_THREAD_ELAPSED:
            return "ULT/elapsed";
        case ABTXI_PROF_WU_TIME_THREAD_CREATE_FIRST_RUN:
            return "ULT/T_firstrun-T_create";
        case ABTXI_PROF_WU_TIME_THREAD_FIRST_RUN_LAST_FINISH:
            return "ULT/T_lastfinish-T_firstrun";
        case ABTXI_PROF_WU_TIME_THREAD_CREATE_LAST_FINISH:
            return "ULT/T_lastfinish-T_create";
        case ABTXI_PROF_WU_TIME_THREAD_CREATE_FREE:
            return "ULT/T_free-T_create";
        case ABTXI_PROF_WU_TIME_TASK_ELAPSED:
            return "tasklet/elapsed";
        case ABTXI_PROF_WU_TIME_TASK_CREATE_FIRST_RUN:
            return "tasklet/T_firstrun-T_create";
        case ABTXI_PROF_WU_TIME_TASK_FIRST_RUN_LAST_FINISH:
            return "tasklet/T_lastfinish-T_firstrun";
        case ABTXI_PROF_WU_TIME_TASK_CREATE_LAST_FINISH:
            return "tasklet/T_lastfinish-T_create";
        case ABTXI_PROF_WU_TIME_TASK_CREATE_FREE:
            return "tasklet/T_free-T_create";
        default:
            return "ERR";
    }
}

#if ABTXI_PROF_USE_TIME_LOCAL
static const char *ABTXI_get_prof_wu_local_time_name(int wu_local_time)
{
    switch (wu_local_time) {
        case ABTXI_PROF_WU_LOCAL_TIME_THREAD_ELAPSED:
            return "ULT/actual_elapsed";
        case ABTXI_PROF_WU_LOCAL_TIME_TASK_ELAPSED:
            return "tasklet/actual_elapsed";
        default:
            return "ERR";
    }
}
#endif

static const char *ABTXI_get_prof_wu_count_name(int wu_count)
{
    switch (wu_count) {
        case ABTXI_PROF_WU_COUNT_THREAD_NUM_REVIVALS:
            return "ULT/revive";
        case ABTXI_PROF_WU_COUNT_THREAD_NUM_YIELDS:
            return "ULT/yield";
        case ABTXI_PROF_WU_COUNT_THREAD_NUM_SUSPENSIONS:
            return "ULT/suspend";
        case ABTXI_PROF_WU_COUNT_THREAD_NUM_XSTREAM_CHANGES:
            return "ULT/ES-change";
        case ABTXI_PROF_WU_COUNT_TASK_NUM_REVIVALS:
            return "tasklet/revive";
        default:
            return "ERR";
    }
}

#define ABTXI_prof_max(a, b) ((a) > (b) ? (a) : (b))
#define ABTXI_prof_min(a, b) ((a) < (b) ? (a) : (b))
#define ABTXI_prof_div_s(a, b) ((b) != 0 ? ((a) / (b)) : 0)
static inline double ABTXI_prof_pow(double base, int exp)
{
    if (exp == 0) {
        return 1.0;
    } else if (exp <= -1) {
        return 1.0 / ABTXI_prof_pow(base, -exp);
    } else {
        double val = ABTXI_prof_pow(base, exp / 2);
        return val * val * (exp % 2 == 0 ? 1.0 : base);
    }
}

static inline int ABTXI_prof_digit(double val)
{
    if (-1.0e-10 < val && val < 1.0e-10) {
        /* Too small.  This is zero. */
        return -99;
    } else if (-1.0 < val && val < 1.0) {
        return -1 + ABTXI_prof_digit(val * 10.0);
    } else if (val < -10.0 || 10.0 < val) {
        return 1 + ABTXI_prof_digit(val * 0.1);
    } else {
        return 0;
    }
}

typedef struct ABTXI_prof_wu_time ABTXI_prof_wu_time;
typedef struct ABTXI_prof_wu_count ABTXI_prof_wu_count;
#if ABTXI_PROF_USE_TIME_LOCAL
typedef struct ABTXI_prof_wu_local_time ABTXI_prof_wu_local_time;
#endif
typedef struct ABTXI_prof_thread_info ABTXI_prof_thread_info;
typedef struct ABTXI_prof_thread_data ABTXI_prof_thread_data;
typedef struct ABTXI_prof_task_info ABTXI_prof_task_info;
typedef struct ABTXI_prof_task_data ABTXI_prof_task_data;
typedef struct ABTXI_prof_xstream_data ABTXI_prof_xstream_data;
typedef struct ABTXI_prof_xstream_info ABTXI_prof_xstream_info;
typedef struct ABTXI_prof_global ABTXI_prof_global;
typedef struct ABTXI_prof_data_table ABTXI_prof_data_table;
typedef struct ABTXI_prof_str_mem ABTXI_prof_str_mem;
typedef struct ABTXI_prof_spinlock ABTXI_prof_spinlock;

struct ABTXI_prof_spinlock {
    volatile int val;
};

struct ABTXI_prof_wu_time {
    ABTXI_PROF_T max_val, min_val, sum;
    uint64_t cnt;
};

struct ABTXI_prof_wu_count {
    uint64_t max_val, min_val, sum;
    uint64_t cnt;
};

#if ABTXI_PROF_USE_TIME_LOCAL
struct ABTXI_prof_wu_local_time {
    ABTXI_PROF_LOCAL_T max_val, min_val, sum;
    uint64_t cnt;
};
#endif

struct ABTXI_prof_thread_data {
    int num_revivals;
    int num_yields;
    int num_suspensions;
    int num_xstream_changes; /* At least one if it runs on only one xstream */
    ABTXI_PROF_T time_created;
    ABTXI_PROF_T time_first_run;
    ABTXI_PROF_T time_last_run;
#if ABTXI_PROF_USE_TIME_LOCAL
    ABTXI_PROF_LOCAL_T time_last_run_local;
#endif
    ABTXI_PROF_T time_last_finish;
    ABTXI_PROF_T time_elapsed;
#if ABTXI_PROF_USE_TIME_LOCAL
    ABTXI_PROF_LOCAL_T time_elapsed_local;
#endif
    ABT_xstream prev_xstream;
    ABT_thread owner;
};

struct ABTXI_prof_thread_info {
    ABTXI_prof_thread_data d;
    ABTXI_prof_thread_info *p_next_unused; /* p_thread_unused. */
    ABTXI_prof_thread_info *p_next_all;    /* p_thread_all */
};

struct ABTXI_prof_task_data {
    int num_revivals;
    ABTXI_PROF_T time_created;
    ABTXI_PROF_T time_first_run;
    ABTXI_PROF_T time_last_run;
#if ABTXI_PROF_USE_TIME_LOCAL
    ABTXI_PROF_LOCAL_T time_last_run_local;
#endif
    ABTXI_PROF_T time_last_finish;
    ABTXI_PROF_T time_elapsed;
#if ABTXI_PROF_USE_TIME_LOCAL
    ABTXI_PROF_LOCAL_T time_elapsed_local;
#endif
    ABT_task owner;
};

struct ABTXI_prof_task_info {
    ABTXI_prof_task_data d;
    ABTXI_prof_task_info *p_next_unused; /* p_task_unused */
    ABTXI_prof_task_info *p_next_all;    /* p_task_all */
};

struct ABTXI_prof_xstream_data {
    int cur_depth; /* Stack depth value "+1" (0: uninitialized) */
    uint64_t num_events[ABTXI_PROF_EVENT_END_];
#if !ABTXI_PROF_USE_TIME_LOCAL
    ABTXI_PROF_T times_last_run[ABTXI_PROF_MAX_DEPTH];
#else
    /* First and last values of ABTXI_prof_get_time_local() */
    ABTXI_PROF_LOCAL_T time_first_run_local;
    ABTXI_PROF_LOCAL_T time_last_run_local;
    ABTXI_PROF_LOCAL_T times_last_run_local[ABTXI_PROF_MAX_DEPTH];
#endif
#if !ABTXI_PROF_USE_TIME_LOCAL
    ABTXI_PROF_T times_elapsed[ABTXI_PROF_MAX_DEPTH]; /* index = depth */
#else
    ABTXI_PROF_LOCAL_T times_elapsed_local[ABTXI_PROF_MAX_DEPTH];
#endif
    ABTXI_prof_wu_time wu_times[ABTXI_PROF_WU_TIME_END_];
    ABTXI_prof_wu_count wu_counts[ABTXI_PROF_WU_COUNT_END_];
#if ABTXI_PROF_USE_TIME_LOCAL
    ABTXI_prof_wu_local_time wu_local_times[ABTXI_PROF_WU_LOCAL_TIME_END_];
#endif
};

struct ABTXI_prof_xstream_info {
    int rank; /* -1 if external threads */
    int tag;  /* Tag for consistency.  Odd value means dirty. */
    ABTXI_prof_xstream_data d;
    /* Memory pool */
    ABTXI_prof_thread_info *p_thread_all;
    ABTXI_prof_thread_info *p_thread_unused;
    ABTXI_prof_task_info *p_task_all;
    ABTXI_prof_task_info *p_task_unused;
    void *p_memblock_head; /* List of memory blocks */
    ABTXI_prof_global *p_global;
    ABTXI_prof_xstream_info *p_next;
};

struct ABTXI_prof_data_table {
    int num_columns;
    const char **column_names;
    int num_rows;
    const char **row_names;
    double *values; /* [row * num_cols + col] */
};

struct ABTXI_prof_str_mem {
    int len, cursor;
    char *s;
    ABTXI_prof_str_mem *p_next;
};

#define ABTXI_PROF_GLOBAL_STATE_CLEAN 0
#define ABTXI_PROF_GLOBAL_STATE_RUNNING 1
#define ABTXI_PROF_GLOBAL_STATE_STOPPED 2

struct ABTXI_prof_global {
    ABT_key prof_key;
    int prof_mode;
    double to_sec; /* ABTXI_PROF_T -> double [s] */
    int state;     /* 0: clean, 1: running, 2: stopped */
    ABTXI_PROF_T start_prof_time;
    ABTXI_PROF_T stop_prof_time;
    /* xstream_info_key is for external threads. */
    pthread_key_t xstream_info_key;
    ABTXI_prof_spinlock xstreams_lock; /* spinlock */
    ABTXI_prof_xstream_info *p_xstream_info_head;
    int len_p_xstreams;                   /* Length of p_xstreams. */
    ABTXI_prof_xstream_info **p_xstreams; /* Can be referenced by "rank" */
    void *mem_p_xstreams;                 /* Memory list of p_xstreams */
};

static inline void ABTXI_prof_spin_init(ABTXI_prof_spinlock *p_lock)
{
#ifndef ABTXI_PROF_USE_SYNC_BUILTIN
    __atomic_clear(&p_lock->val, __ATOMIC_RELAXED);
#else
    p_lock->val = 0;
#endif
}

static inline void ABTXI_prof_spin_destroy(ABTXI_prof_spinlock *p_lock)
{
#ifndef ABTXI_PROF_USE_SYNC_BUILTIN
    __atomic_clear(&p_lock->val, __ATOMIC_RELAXED);
#else
    p_lock->val = 0;
#endif
}

static inline void ABTXI_prof_spin_lock(ABTXI_prof_spinlock *p_lock)
{
#ifndef ABTXI_PROF_USE_SYNC_BUILTIN
    while (__atomic_test_and_set(&p_lock->val, __ATOMIC_ACQUIRE))
        ;
#else
    while (__sync_lock_test_and_set(&p_lock->val, 1))
        __sync_synchronize();
#endif
}

static inline void ABTXI_prof_spin_unlock(ABTXI_prof_spinlock *p_lock)
{
#ifndef ABTXI_PROF_USE_SYNC_BUILTIN
    __atomic_clear(&p_lock->val, __ATOMIC_RELEASE);
#else
    __sync_lock_release(&p_lock->val);
#endif
}

static inline int ABTXI_prof_atomic_relaxed_load_int(int *p_int)
{
#ifndef ABTXI_PROF_USE_SYNC_BUILTIN
    return __atomic_load_n(p_int, __ATOMIC_RELAXED);
#else
    return *((volatile int *)p_int);
#endif
}

static inline int ABTXI_prof_atomic_acquire_load_int(int *p_int)
{
#ifndef ABTXI_PROF_USE_SYNC_BUILTIN
    return __atomic_load_n(p_int, __ATOMIC_RELAXED);
#else
    int ret;
    __sync_synchronize();
    ret = *((volatile int *)p_int);
    __sync_synchronize();
    return ret;
#endif
}

static inline void ABTXI_prof_atomic_relaxed_store_int(int *p_int, int val)
{
#ifndef ABTXI_PROF_USE_SYNC_BUILTIN
    __atomic_store_n(p_int, val, __ATOMIC_RELAXED);
#else
    *((volatile int *)p_int) = val;
#endif
}

static inline void ABTXI_prof_atomic_release_store_int(int *p_int, int val)
{
#ifndef ABTXI_PROF_USE_SYNC_BUILTIN
    __atomic_store_n(p_int, val, __ATOMIC_RELEASE);
#else
    __sync_synchronize();
    *((volatile int *)p_int) = val;
    __sync_synchronize();
#endif
}

static inline void *ABTXI_prof_atomic_relaxed_load_ptr(void **p_ptr)
{
#ifndef ABTXI_PROF_USE_SYNC_BUILTIN
    return __atomic_load_n(p_ptr, __ATOMIC_RELAXED);
#else
    return *((void *volatile *)p_ptr);
#endif
}

static inline void *ABTXI_prof_atomic_acquire_load_ptr(void **p_ptr)
{
#ifndef ABTXI_PROF_USE_SYNC_BUILTIN
    return __atomic_load_n(p_ptr, __ATOMIC_RELAXED);
#else
    void *ret;
    __sync_synchronize();
    ret = *((void *volatile *)p_ptr);
    __sync_synchronize();
    return ret;
#endif
}

static inline void ABTXI_prof_atomic_relaxed_store_ptr(void **p_ptr, void *val)
{
#ifndef ABTXI_PROF_USE_SYNC_BUILTIN
    __atomic_store_n(p_ptr, val, __ATOMIC_RELAXED);
#else
    *((void *volatile *)p_ptr) = val;
#endif
}

static inline void ABTXI_prof_atomic_release_store_ptr(void **p_ptr, void *val)
{
#ifndef ABTXI_PROF_USE_SYNC_BUILTIN
    __atomic_store_n(p_ptr, val, __ATOMIC_RELEASE);
#else
    __sync_synchronize();
    *((void *volatile *)p_ptr) = val;
    __sync_synchronize();
#endif
}

static inline void ABTXI_prof_wu_time_add(ABTXI_prof_wu_time *p_wu_time,
                                          ABTXI_PROF_T val)
{
    if (ABTXI_prof_unlikely(p_wu_time->cnt == 0)) {
        p_wu_time->max_val = val;
        p_wu_time->min_val = val;
    } else {
        if (p_wu_time->max_val < val)
            p_wu_time->max_val = val;
        if (p_wu_time->min_val > val)
            p_wu_time->min_val = val;
    }
    p_wu_time->sum += val;
    p_wu_time->cnt += 1;
}

static void ABTXI_prof_wu_time_merge(ABTXI_prof_wu_time *p_dest,
                                     const ABTXI_prof_wu_time *p_src)
{
    if (p_dest->cnt == 0) {
        p_dest->max_val = p_src->max_val;
        p_dest->min_val = p_src->min_val;
    } else if (p_src->cnt != 0) {
        if (p_dest->max_val < p_src->max_val)
            p_dest->max_val = p_src->max_val;
        if (p_dest->min_val > p_src->min_val)
            p_dest->min_val = p_src->min_val;
    }
    p_dest->sum += p_src->sum;
    p_dest->cnt += p_src->cnt;
}

static inline void ABTXI_prof_wu_count_add(ABTXI_prof_wu_count *p_wu_count,
                                           uint64_t val)
{
    if (ABTXI_prof_unlikely(p_wu_count->cnt == 0)) {
        p_wu_count->max_val = val;
        p_wu_count->min_val = val;
    } else {
        if (p_wu_count->max_val < val)
            p_wu_count->max_val = val;
        if (p_wu_count->min_val > val)
            p_wu_count->min_val = val;
    }
    p_wu_count->sum += val;
    p_wu_count->cnt += 1;
}

static void ABTXI_prof_wu_count_merge(ABTXI_prof_wu_count *p_dest,
                                      const ABTXI_prof_wu_count *p_src)
{
    if (p_dest->cnt == 0) {
        p_dest->max_val = p_src->max_val;
        p_dest->min_val = p_src->min_val;
    } else if (p_src->cnt != 0) {
        if (p_dest->max_val < p_src->max_val)
            p_dest->max_val = p_src->max_val;
        if (p_dest->min_val > p_src->min_val)
            p_dest->min_val = p_src->min_val;
    }
    p_dest->sum += p_src->sum;
    p_dest->cnt += p_src->cnt;
}

#if ABTXI_PROF_USE_TIME_LOCAL
static inline void
ABTXI_prof_wu_local_time_add(ABTXI_prof_wu_local_time *p_wu_local_time,
                             ABTXI_PROF_LOCAL_T val)
{
    if (ABTXI_prof_unlikely(p_wu_local_time->cnt == 0)) {
        p_wu_local_time->max_val = val;
        p_wu_local_time->min_val = val;
    } else {
        if (p_wu_local_time->max_val < val)
            p_wu_local_time->max_val = val;
        if (p_wu_local_time->min_val > val)
            p_wu_local_time->min_val = val;
    }
    p_wu_local_time->sum += val;
    p_wu_local_time->cnt += 1;
}

static void
ABTXI_prof_wu_local_time_merge(ABTXI_prof_wu_local_time *p_dest,
                               const ABTXI_prof_wu_local_time *p_src)
{
    if (p_dest->cnt == 0) {
        p_dest->max_val = p_src->max_val;
        p_dest->min_val = p_src->min_val;
    } else if (p_src->cnt != 0) {
        if (p_dest->max_val < p_src->max_val)
            p_dest->max_val = p_src->max_val;
        if (p_dest->min_val > p_src->min_val)
            p_dest->min_val = p_src->min_val;
    }
    p_dest->sum += p_src->sum;
    p_dest->cnt += p_src->cnt;
}
#endif

static ABTXI_prof_str_mem *ABTXI_prof_str_mem_alloc(int reserved)
{
    ABTXI_prof_str_mem *p_str =
        (ABTXI_prof_str_mem *)malloc(sizeof(ABTXI_prof_str_mem) + reserved);
    p_str->s = ((char *)p_str) + sizeof(ABTXI_prof_str_mem);
    p_str->len = reserved;
    p_str->cursor = 0;
    p_str->p_next = 0;
    return p_str;
}

static char *ABTXI_prof_sprintf(ABTXI_prof_str_mem *p_str, size_t max_n,
                                const char *format, ...)
{
    va_list args;
    va_start(args, format);
    while (p_str->p_next) {
        p_str = p_str->p_next;
    }
    if (p_str->len - p_str->cursor < max_n) {
        int newlen = max_n > 4096 ? max_n : 4096;
        ABTXI_prof_str_mem *p_new = ABTXI_prof_str_mem_alloc(newlen);
        p_str->p_next = p_new;
        p_str = p_new;
    }
    char *s = p_str->s + p_str->cursor;
    int len = vsprintf(s, format, args);
    p_str->cursor += len + 1;
    va_end(args);
    return s;
}

static void ABTXI_prof_str_mem_free(ABTXI_prof_str_mem *p_str)
{
    while (p_str) {
        ABTXI_prof_str_mem *p_next = p_str->p_next;
        free(p_str);
        p_str = p_next;
    }
}

static void ABTXI_prof_xstream_info_alloc_thread_info(
    ABTXI_prof_xstream_info *p_xstream_info)
{
    void *p_memblock = calloc(1, ABTXI_PROF_MEM_BLOCK_SIZE);
    /* Add the newly allocate memblock to p_memblock_head */
    *(void **)p_memblock = p_xstream_info->p_memblock_head;
    p_xstream_info->p_memblock_head = p_memblock;
    /* Extract empty thread_info from memblock. */
    size_t offset = 128; /* 128 bytes for safe alignment.  Note that the first
                          * block contains a pointer to the next memblock  */
    ABTXI_prof_thread_info *p_head_unused = p_xstream_info->p_thread_unused;
    ABTXI_prof_thread_info *p_head_all = p_xstream_info->p_thread_all;
    while (offset + sizeof(ABTXI_prof_thread_info) <=
           ABTXI_PROF_MEM_BLOCK_SIZE) {
        ABTXI_prof_thread_info *p_new =
            (ABTXI_prof_thread_info *)(((char *)p_memblock) + offset);
        p_new->p_next_unused = p_head_unused;
        p_new->p_next_all = p_head_all;
        p_head_unused = p_new;
        p_head_all = p_new;
        offset += sizeof(ABTXI_prof_thread_info);
    }
    p_xstream_info->p_thread_unused = p_head_unused;
    /* p_thread_all must be updated atomically since it might be read by a print
     * thread asynchronously. */
    ABTXI_prof_atomic_release_store_ptr((void **)&p_xstream_info->p_thread_all,
                                        (void *)p_head_all);
}

static void
ABTXI_prof_xstream_info_alloc_task_info(ABTXI_prof_xstream_info *p_xstream_info)
{
    void *p_memblock = calloc(1, ABTXI_PROF_MEM_BLOCK_SIZE);
    /* Add the newly allocate memblock to p_memblock_head */
    *(void **)p_memblock = p_xstream_info->p_memblock_head;
    p_xstream_info->p_memblock_head = p_memblock;
    /* Extract empty task_info from memblock. */
    size_t offset = 128; /* 128 bytes for safe alignment.  Note that the first
                          * block contains a pointer to the next memblock  */
    ABTXI_prof_task_info *p_head_unused = p_xstream_info->p_task_unused;
    ABTXI_prof_task_info *p_head_all = p_xstream_info->p_task_all;
    while (offset + sizeof(ABTXI_prof_task_info) <= ABTXI_PROF_MEM_BLOCK_SIZE) {
        ABTXI_prof_task_info *p_new =
            (ABTXI_prof_task_info *)(((char *)p_memblock) + offset);
        p_new->p_next_unused = p_head_unused;
        p_new->p_next_all = p_head_all;
        p_head_unused = p_new;
        p_head_all = p_new;
        offset += sizeof(ABTXI_prof_task_info);
    }
    p_xstream_info->p_task_unused = p_head_unused;
    /* p_task_all must be updated atomically since it might be read by a print
     * thread asynchronously. */
    ABTXI_prof_atomic_release_store_ptr((void **)&p_xstream_info->p_task_all,
                                        (void *)p_head_all);
}

static inline void
ABTXI_prof_init_thread_info(ABTXI_prof_thread_info *p_thread_info)
{
    /* Zero clear. */
    memset(&p_thread_info->d, 0, sizeof(ABTXI_prof_thread_data));
}

static inline void ABTXI_prof_init_task_info(ABTXI_prof_task_info *p_task_info)
{
    /* Zero clear. */
    memset(&p_task_info->d, 0, sizeof(ABTXI_prof_task_data));
}

static void ABTXI_prof_reset_thread_info(ABTXI_prof_thread_info *p_thread_info)
{
    /* Basically zero clear. */
    memset(&p_thread_info->d, 0, sizeof(ABTXI_prof_thread_data));
    p_thread_info->p_next_unused = p_thread_info->p_next_all;
}

static void ABTXI_prof_reset_task_info(ABTXI_prof_task_info *p_task_info)
{
    /* Basically zero clear. */
    memset(&p_task_info->d, 0, sizeof(ABTXI_prof_task_data));
    p_task_info->p_next_unused = p_task_info->p_next_all;
}

static inline ABTXI_prof_thread_info *
ABTXI_prof_get_thread_info(ABTXI_prof_global *p_global,
                           ABTXI_prof_xstream_info *p_xstream_info,
                           ABT_thread thread)
{
    /* Multiple thread events will not be invoked for the same thread. */
    ABTXI_prof_thread_info *p_thread_info;
    ABT_key prof_key = p_global->prof_key;
    ABT_thread_get_specific(thread, prof_key, (void **)&p_thread_info);
    /* owner can be changed if thread_info has been reset by restarting the
     * profiler.  If it is the case, this p_thread_info is no longer belonging
     * to this thread, so a new one must be allocated. */
    if (ABTXI_prof_likely(p_thread_info && p_thread_info->d.owner == thread)) {
        return p_thread_info;
    } else {
        if (!p_xstream_info->p_thread_unused) {
            ABTXI_prof_xstream_info_alloc_thread_info(p_xstream_info);
        }
        p_thread_info = p_xstream_info->p_thread_unused;
        /* This p_thread_info has been already initialized. */
        p_xstream_info->p_thread_unused = p_thread_info->p_next_unused;
        ABT_thread_set_specific(thread, prof_key, (void *)p_thread_info);
        p_thread_info->d.owner = thread;
        return p_thread_info;
    }
}

static inline ABTXI_prof_task_info *
ABTXI_prof_get_task_info(ABTXI_prof_global *p_global,
                         ABTXI_prof_xstream_info *p_xstream_info, ABT_task task)
{
    /* Multiple task events will not be invoked for the same task. */
    ABTXI_prof_task_info *p_task_info;
    ABT_key prof_key = p_global->prof_key;
    ABT_task_get_specific(task, prof_key, (void **)&p_task_info);
    /* owner can be changed if task_info has been reset by restarting the
     * profiler.  If it is the case, this p_task_info is no longer belonging to
     * this task, so a new one must be allocated. */
    if (ABTXI_prof_likely(p_task_info && p_task_info->d.owner == task)) {
        return p_task_info;
    } else {
        if (!p_xstream_info->p_task_unused) {
            ABTXI_prof_xstream_info_alloc_task_info(p_xstream_info);
        }
        p_task_info = p_xstream_info->p_task_unused;
        /* This p_task_info has been already initialized. */
        p_xstream_info->p_task_unused = p_task_info->p_next_unused;
        ABT_task_set_specific(task, prof_key, (void *)p_task_info);
        p_task_info->d.owner = task;
        return p_task_info;
    }
}

static inline void
ABTXI_prof_merge_thread_data(ABTXI_prof_xstream_data *p_xstream_data,
                             const ABTXI_prof_thread_data *p_thread_data,
                             ABTXI_PROF_T time_freed)
{
    /* Update statistics (counts) */
    ABTXI_prof_wu_count_add(&p_xstream_data->wu_counts
                                 [ABTXI_PROF_WU_COUNT_THREAD_NUM_REVIVALS],
                            p_thread_data->num_revivals);
    ABTXI_prof_wu_count_add(&p_xstream_data->wu_counts
                                 [ABTXI_PROF_WU_COUNT_THREAD_NUM_YIELDS],
                            p_thread_data->num_yields);
    ABTXI_prof_wu_count_add(&p_xstream_data->wu_counts
                                 [ABTXI_PROF_WU_COUNT_THREAD_NUM_SUSPENSIONS],
                            p_thread_data->num_suspensions);
    ABTXI_prof_wu_count_add(
        &p_xstream_data
             ->wu_counts[ABTXI_PROF_WU_COUNT_THREAD_NUM_XSTREAM_CHANGES],
        p_thread_data->num_xstream_changes);
    /* Update statistics (times) */
    ABTXI_PROF_T time_created = p_thread_data->time_created;
    ABTXI_PROF_T time_first_run = p_thread_data->time_first_run;
    ABTXI_PROF_T time_last_finish = p_thread_data->time_last_finish;
    ABTXI_PROF_T time_elapsed = p_thread_data->time_elapsed;
    if (ABTXI_prof_likely(time_elapsed != ABTXI_PROF_T_ZERO)) {
        ABTXI_prof_wu_time_add(&p_xstream_data->wu_times
                                    [ABTXI_PROF_WU_TIME_THREAD_ELAPSED],
                               time_elapsed);
    }
    if (ABTXI_prof_likely(time_created != ABTXI_PROF_T_ZERO &&
                          time_first_run != ABTXI_PROF_T_ZERO)) {
        ABTXI_prof_wu_time_add(
            &p_xstream_data
                 ->wu_times[ABTXI_PROF_WU_TIME_THREAD_CREATE_FIRST_RUN],
            time_first_run - time_created);
    }
    if (ABTXI_prof_likely(time_first_run != ABTXI_PROF_T_ZERO &&
                          time_last_finish != ABTXI_PROF_T_ZERO)) {
        ABTXI_prof_wu_time_add(
            &p_xstream_data
                 ->wu_times[ABTXI_PROF_WU_TIME_THREAD_FIRST_RUN_LAST_FINISH],
            time_last_finish - time_first_run);
    }
    if (ABTXI_prof_likely(time_created != ABTXI_PROF_T_ZERO &&
                          time_last_finish != ABTXI_PROF_T_ZERO)) {
        ABTXI_prof_wu_time_add(
            &p_xstream_data
                 ->wu_times[ABTXI_PROF_WU_TIME_THREAD_CREATE_LAST_FINISH],
            time_last_finish - time_created);
    }
    if (ABTXI_prof_likely(time_created != ABTXI_PROF_T_ZERO &&
                          time_freed != ABTXI_PROF_T_ZERO)) {
        ABTXI_prof_wu_time_add(&p_xstream_data->wu_times
                                    [ABTXI_PROF_WU_TIME_THREAD_CREATE_FREE],
                               time_freed - time_created);
    }
#if ABTXI_PROF_USE_TIME_LOCAL
    /* Update statistics (local times) */
    ABTXI_PROF_LOCAL_T time_elapsed_local = p_thread_data->time_elapsed_local;
    if (ABTXI_prof_likely(time_elapsed_local != ABTXI_PROF_LOCAL_T_ZERO)) {
        ABTXI_prof_wu_local_time_add(
            &p_xstream_data
                 ->wu_local_times[ABTXI_PROF_WU_LOCAL_TIME_THREAD_ELAPSED],
            time_elapsed_local);
    }
#endif
}

static inline void
ABTXI_prof_merge_task_data(ABTXI_prof_xstream_data *p_xstream_data,
                           const ABTXI_prof_task_data *p_task_data,
                           ABTXI_PROF_T time_freed)
{
    /* Update statistics (counts) */
    ABTXI_prof_wu_count_add(&p_xstream_data->wu_counts
                                 [ABTXI_PROF_WU_COUNT_TASK_NUM_REVIVALS],
                            p_task_data->num_revivals);
    /* Update statistics (times) */
    ABTXI_PROF_T time_created = p_task_data->time_created;
    ABTXI_PROF_T time_first_run = p_task_data->time_first_run;
    ABTXI_PROF_T time_last_finish = p_task_data->time_last_finish;
    ABTXI_PROF_T time_elapsed = p_task_data->time_elapsed;
    if (ABTXI_prof_likely(time_elapsed != ABTXI_PROF_T_ZERO)) {
        ABTXI_prof_wu_time_add(&p_xstream_data
                                    ->wu_times[ABTXI_PROF_WU_TIME_TASK_ELAPSED],
                               time_elapsed);
    }
    if (ABTXI_prof_likely(time_created != ABTXI_PROF_T_ZERO &&
                          time_first_run != ABTXI_PROF_T_ZERO)) {
        ABTXI_prof_wu_time_add(&p_xstream_data->wu_times
                                    [ABTXI_PROF_WU_TIME_TASK_CREATE_FIRST_RUN],
                               time_first_run - time_created);
    }
    if (ABTXI_prof_likely(time_first_run != ABTXI_PROF_T_ZERO &&
                          time_last_finish != ABTXI_PROF_T_ZERO)) {
        ABTXI_prof_wu_time_add(
            &p_xstream_data
                 ->wu_times[ABTXI_PROF_WU_TIME_TASK_FIRST_RUN_LAST_FINISH],
            time_last_finish - time_first_run);
    }
    if (ABTXI_prof_likely(time_created != ABTXI_PROF_T_ZERO &&
                          time_last_finish != ABTXI_PROF_T_ZERO)) {
        ABTXI_prof_wu_time_add(
            &p_xstream_data
                 ->wu_times[ABTXI_PROF_WU_TIME_TASK_CREATE_LAST_FINISH],
            time_last_finish - time_created);
    }
    if (ABTXI_prof_likely(time_created != ABTXI_PROF_T_ZERO &&
                          time_freed != ABTXI_PROF_T_ZERO)) {
        ABTXI_prof_wu_time_add(&p_xstream_data->wu_times
                                    [ABTXI_PROF_WU_TIME_TASK_CREATE_FREE],
                               time_freed - time_created);
    }
#if ABTXI_PROF_USE_TIME_LOCAL
    /* Update statistics (local times) */
    ABTXI_PROF_LOCAL_T time_elapsed_local = p_task_data->time_elapsed_local;
    if (ABTXI_prof_likely(time_elapsed_local != ABTXI_PROF_LOCAL_T_ZERO)) {
        ABTXI_prof_wu_local_time_add(
            &p_xstream_data
                 ->wu_local_times[ABTXI_PROF_WU_LOCAL_TIME_TASK_ELAPSED],
            time_elapsed_local);
    }
#endif
}

static inline void
ABTXI_prof_release_thread_info(ABTXI_prof_xstream_info *p_xstream_info,
                               ABTXI_prof_thread_info *p_thread_info,
                               ABTXI_PROF_T time_freed)
{
    /* Update statistics */
    ABTXI_prof_merge_thread_data(&p_xstream_info->d, &p_thread_info->d,
                                 time_freed);
    /* Return to the memory pool. */
    ABTXI_prof_init_thread_info(p_thread_info);
    p_thread_info->p_next_unused = p_xstream_info->p_thread_unused;
    p_xstream_info->p_thread_unused = p_thread_info;
}

static inline void
ABTXI_prof_release_task_info(ABTXI_prof_xstream_info *p_xstream_info,
                             ABTXI_prof_task_info *p_task_info,
                             ABTXI_PROF_T time_freed)
{
    /* Update statistics */
    ABTXI_prof_merge_task_data(&p_xstream_info->d, &p_task_info->d, time_freed);
    /* Return to the memory pool. */
    ABTXI_prof_init_task_info(p_task_info);
    p_task_info->p_next_unused = p_xstream_info->p_task_unused;
    p_xstream_info->p_task_unused = p_task_info;
}

static void
ABTXI_prof_init_xstream_info(ABTXI_prof_global *p_global,
                             ABTXI_prof_xstream_info *p_xstream_info, int rank)
{
    /* Zero clear. */
    memset(p_xstream_info, 0, sizeof(ABTXI_prof_xstream_info));
    p_xstream_info->p_global = p_global;
    p_xstream_info->rank = rank;
}

static void
ABTXI_prof_reset_xstream_info(ABTXI_prof_xstream_info *p_xstream_info)
{
    /* Basically zero clear. */
    ABTXI_prof_thread_info *p_thread_all = p_xstream_info->p_thread_all;
    ABTXI_prof_task_info *p_task_all = p_xstream_info->p_task_all;
    memset(&p_xstream_info->d, 0, sizeof(ABTXI_prof_xstream_data));

    /* Reset thread_info */
    ABTXI_prof_thread_info *p_thread_cur = p_thread_all;
    while (p_thread_cur) {
        ABTXI_prof_reset_thread_info(p_thread_cur);
        p_thread_cur = p_thread_cur->p_next_all;
    }
    p_xstream_info->p_thread_unused = p_thread_all;

    /* Reset task_info */
    ABTXI_prof_task_info *p_task_cur = p_task_all;
    while (p_task_cur) {
        ABTXI_prof_reset_task_info(p_task_cur);
        p_task_cur = p_task_cur->p_next_all;
    }
    p_xstream_info->p_task_unused = p_task_all;
}

static void
ABTXI_prof_destroy_xstream_info(ABTXI_prof_xstream_info *p_xstream_info)
{
    /* Memory blocks must be freed when all the xstream_infos are freed. */
    void *p_head = p_xstream_info->p_memblock_head;
    while (p_head) {
        void *p_next = *(void **)p_head;
        free(p_head);
        p_head = p_next;
    }
}

static inline ABTXI_prof_xstream_info *
ABTXI_prof_get_xstream_info(ABTXI_prof_global *p_global, ABT_xstream xstream)
{
    if (ABTXI_prof_unlikely(xstream == ABT_XSTREAM_NULL)) {
        ABTXI_prof_xstream_info *p_xstream_info =
            (ABTXI_prof_xstream_info *)pthread_getspecific(
                p_global->xstream_info_key);
        if (ABTXI_prof_unlikely(!p_xstream_info)) {
            /* The first time of creation. */
            p_xstream_info = (ABTXI_prof_xstream_info *)malloc(
                sizeof(ABTXI_prof_xstream_info));
            ABTXI_prof_init_xstream_info(p_global, p_xstream_info, -1);
            int ret = pthread_setspecific(p_global->xstream_info_key,
                                          (void *)p_xstream_info);
            assert(ret == 0);
            ABTXI_prof_spin_lock(&p_global->xstreams_lock);
            /* Add to the linked list. */
            p_xstream_info->p_next = p_global->p_xstream_info_head;
            p_global->p_xstream_info_head = p_xstream_info;
            ABTXI_prof_spin_unlock(&p_global->xstreams_lock);
        }
        return p_xstream_info;
    } else {
        /* Use a rank to get the associated xstream_info */
        int rank;
        ABT_xstream_get_rank(xstream, &rank);
        if (ABTXI_prof_unlikely(ABTXI_prof_atomic_acquire_load_int(
                                    &p_global->len_p_xstreams) <= rank)) {
            /* Extend the array without synchronizing readers. */
            ABTXI_prof_spin_lock(&p_global->xstreams_lock);
            int len_p_xstreams =
                ABTXI_prof_atomic_acquire_load_int(&p_global->len_p_xstreams);
            if (len_p_xstreams <= rank) {
                int new_len_p_xstreams =
                    ABTXI_prof_max(ABTXI_PROF_DEFAULT_NUM_XSTREAMS,
                                   len_p_xstreams * 2);
                size_t size_mem_p_xstreams_p_next = sizeof(void *);
                size_t size_p_xstreams =
                    sizeof(ABTXI_prof_xstream_info *) * new_len_p_xstreams;
                void *p_new_mem =
                    calloc(1, size_mem_p_xstreams_p_next + size_p_xstreams);
                /* Add this new memory */
                *(void **)p_new_mem = p_global->mem_p_xstreams;
                p_global->mem_p_xstreams = p_new_mem;
                /* Copy the current data. */
                ABTXI_prof_xstream_info **p_new_xstreams =
                    (ABTXI_prof_xstream_info **)(((char *)p_new_mem) +
                                                 size_mem_p_xstreams_p_next);
                memcpy(p_new_xstreams, p_global->p_xstreams,
                       sizeof(ABTXI_prof_xstream_info *) * len_p_xstreams);
                ABTXI_prof_atomic_relaxed_store_ptr((void **)&p_global
                                                        ->p_xstreams,
                                                    p_new_xstreams);
                ABTXI_prof_atomic_release_store_int(&p_global->len_p_xstreams,
                                                    new_len_p_xstreams);
            } else {
                /* Another execution stream has already extended this array. */
            }
            ABTXI_prof_spin_unlock(&p_global->xstreams_lock);
        }
        ABTXI_prof_xstream_info **p_xstreams =
            (ABTXI_prof_xstream_info **)ABTXI_prof_atomic_relaxed_load_ptr(
                (void **)&p_global->p_xstreams);
        ABTXI_prof_xstream_info *p_xstream_info =
            (ABTXI_prof_xstream_info *)ABTXI_prof_atomic_relaxed_load_ptr(
                (void **)&p_xstreams[rank]);
        if (ABTXI_prof_unlikely(p_xstream_info == NULL)) {
            /* p_xstream has not been allocated yet.  Allocate and assign it. */
            p_xstream_info = (ABTXI_prof_xstream_info *)malloc(
                sizeof(ABTXI_prof_xstream_info));
            ABTXI_prof_init_xstream_info(p_global, p_xstream_info, rank);
            ABTXI_prof_spin_lock(&p_global->xstreams_lock);
            /* Add to p_xstreams. */
            p_global->p_xstreams[rank] = p_xstream_info;
            /* Add to the linked list. */
            p_xstream_info->p_next = p_global->p_xstream_info_head;
            p_global->p_xstream_info_head = p_xstream_info;
            ABTXI_prof_spin_unlock(&p_global->xstreams_lock);
        }
        return p_xstream_info;
    }
}

static inline int ABTXI_prof_xstream_info_get_depth_thread(
    ABTXI_prof_xstream_info *p_xstream_info, uint64_t event,
    ABT_tool_context context, int force_update)
{
    int cur_depth;
    if (force_update ||
        ABTXI_prof_unlikely(((cur_depth = p_xstream_info->d.cur_depth)) == 0)) {
        int stack_depth = 0;
        ABT_tool_query_thread(context, event, ABT_TOOL_QUERY_KIND_STACK_DEPTH,
                              &stack_depth);
        cur_depth = stack_depth + 1;
        p_xstream_info->d.cur_depth = cur_depth;
    }
    /* cur_depth is the actual depth plus 1 */
    return cur_depth - 1;
}

static inline int
ABTXI_prof_xstream_info_get_depth_task(ABTXI_prof_xstream_info *p_xstream_info,
                                       uint64_t event, ABT_tool_context context,
                                       int force_update)
{
    int cur_depth;
    if (force_update ||
        ABTXI_prof_unlikely(((cur_depth = p_xstream_info->d.cur_depth)) == 0)) {
        int stack_depth = 0;
        ABT_tool_query_task(context, event, ABT_TOOL_QUERY_KIND_STACK_DEPTH,
                            &stack_depth);
        cur_depth = stack_depth + 1;
        p_xstream_info->d.cur_depth = cur_depth;
    }
    /* cur_depth is the actual depth plus 1 */
    return cur_depth - 1;
}

static ABTXI_prof_always_inline void
ABTXI_prof_thread_callback_impl(ABT_thread thread, ABT_xstream xstream,
                                uint64_t event, ABT_tool_context context,
                                void *user_arg, int prof_mode)
{
    ABTXI_prof_global *p_global = (ABTXI_prof_global *)user_arg;
    ABTXI_prof_xstream_info *p_xstream_info =
        ABTXI_prof_get_xstream_info(p_global, xstream);

    /* Enter the dirty phase.  This is needed to avoid the print function
     * reading premature values. */
    ABTXI_prof_atomic_release_store_int(&p_xstream_info->tag,
                                        p_xstream_info->tag + 1);

    ABTXI_prof_thread_info *p_thread_info;
    if (prof_mode == ABTX_PROF_MODE_DETAILED) {
        p_thread_info =
            ABTXI_prof_get_thread_info(p_global, p_xstream_info, thread);
    }
    switch (event) {
        case ABT_TOOL_EVENT_THREAD_CREATE: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_THREAD_CREATE]++;

            if (prof_mode == ABTX_PROF_MODE_DETAILED) {
                ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
                p_thread_info->d.time_created = cur_time;
            }
            break;
        }
        case ABT_TOOL_EVENT_THREAD_JOIN: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_THREAD_JOIN]++;
            break;
        }
        case ABT_TOOL_EVENT_THREAD_FREE: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_THREAD_FREE]++;

            /* thread_info is no longer needed. */
            if (prof_mode == ABTX_PROF_MODE_DETAILED) {
                ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
                ABTXI_prof_release_thread_info(p_xstream_info, p_thread_info,
                                               cur_time);
            }
            break;
        }
        case ABT_TOOL_EVENT_THREAD_REVIVE: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_THREAD_REVIVE]++;

            if (prof_mode == ABTX_PROF_MODE_DETAILED) {
                p_thread_info->d.num_revivals++;
            }
            break;
        }
        case ABT_TOOL_EVENT_THREAD_RUN: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_THREAD_RUN]++;

#if !ABTXI_PROF_USE_TIME_LOCAL
            ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
#else
            ABTXI_PROF_LOCAL_T cur_time_local = ABTXI_prof_get_time_local();
            p_xstream_info->d.time_last_run_local = cur_time_local;
            if (ABTXI_prof_unlikely(p_xstream_info->d.time_first_run_local ==
                                    ABTXI_PROF_LOCAL_T_ZERO)) {
                p_xstream_info->d.time_first_run_local = cur_time_local;
            }
#endif

            int depth =
                ABTXI_prof_xstream_info_get_depth_thread(p_xstream_info, event,
                                                         context, 1);
            if (depth < ABTXI_PROF_MAX_DEPTH) {
#if !ABTXI_PROF_USE_TIME_LOCAL
                p_xstream_info->d.times_last_run[depth] = cur_time;
#else
                p_xstream_info->d.times_last_run_local[depth] = cur_time_local;
#endif
            }

            if (prof_mode == ABTX_PROF_MODE_DETAILED) {
#if ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
#endif
                if (p_thread_info->d.time_first_run == ABTXI_PROF_T_ZERO) {
                    p_thread_info->d.time_first_run = cur_time;
                }
                p_thread_info->d.time_last_run = cur_time;
#if ABTXI_PROF_USE_TIME_LOCAL
                p_thread_info->d.time_last_run_local = cur_time_local;
#endif

                if (p_thread_info->d.prev_xstream != xstream) {
                    p_thread_info->d.num_xstream_changes++;
                }
                p_thread_info->d.prev_xstream = xstream;
            }
            break;
        }
        case ABT_TOOL_EVENT_THREAD_FINISH: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_THREAD_FINISH]++;

#if !ABTXI_PROF_USE_TIME_LOCAL
            ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
#else
            ABTXI_PROF_LOCAL_T cur_time_local = ABTXI_prof_get_time_local();
            p_xstream_info->d.time_last_run_local = cur_time_local;
            if (ABTXI_prof_unlikely(p_xstream_info->d.time_first_run_local ==
                                    ABTXI_PROF_LOCAL_T_ZERO)) {
                p_xstream_info->d.time_first_run_local = cur_time_local;
            }
#endif

            int depth =
                ABTXI_prof_xstream_info_get_depth_thread(p_xstream_info, event,
                                                         context, 0);
            if (depth < ABTXI_PROF_MAX_DEPTH) {
#if !ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_T xstream_time_last_run =
                    p_xstream_info->d.times_last_run[depth];
                if (ABTXI_prof_unlikely(xstream_time_last_run ==
                                        ABTXI_PROF_T_ZERO)) {
                    /* Adjustment */
                    xstream_time_last_run = p_global->start_prof_time;
                    p_xstream_info->d.times_last_run[depth] =
                        xstream_time_last_run;
                }
                p_xstream_info->d.times_elapsed[depth] +=
                    cur_time - xstream_time_last_run;
                p_xstream_info->d.times_last_run[depth] =
                    ABTXI_PROF_TIME_LAST_RUN_INVALID;
#else
                ABTXI_PROF_LOCAL_T xstream_time_last_run_local =
                    p_xstream_info->d.times_last_run_local[depth];
                if (ABTXI_prof_unlikely(xstream_time_last_run_local ==
                                        ABTXI_PROF_LOCAL_T_ZERO)) {
                    /* Adjustment */
                    xstream_time_last_run_local = cur_time_local;
                    p_xstream_info->d.times_last_run_local[depth] =
                        cur_time_local;
                }
                p_xstream_info->d.times_elapsed_local[depth] +=
                    cur_time_local - xstream_time_last_run_local;
                p_xstream_info->d.times_last_run_local[depth] =
                    ABTXI_PROF_TIME_LAST_RUN_LOCAL_INVALID;
#endif
            }

            if (prof_mode == ABTX_PROF_MODE_DETAILED) {
#if ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
#endif
                ABTXI_PROF_T thread_time_last_run =
                    p_thread_info->d.time_last_run;
                if (ABTXI_prof_unlikely(thread_time_last_run ==
                                        ABTXI_PROF_T_ZERO)) {
                    /* Adjustment */
                    thread_time_last_run = p_global->start_prof_time;
                    p_thread_info->d.time_last_run = thread_time_last_run;
                }
                p_thread_info->d.time_elapsed +=
                    cur_time - thread_time_last_run;
#if ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_LOCAL_T thread_time_last_run_local =
                    p_thread_info->d.time_last_run_local;
                if (ABTXI_prof_unlikely(thread_time_last_run_local ==
                                        ABTXI_PROF_LOCAL_T_ZERO)) {
                    /* Adjustment */
                    thread_time_last_run_local = cur_time_local;
                    p_thread_info->d.time_last_run_local = cur_time_local;
                }
                p_thread_info->d.time_elapsed_local +=
                    cur_time_local - thread_time_last_run_local;
#endif
                p_thread_info->d.time_last_finish = cur_time;
            }
            break;
        }
        case ABT_TOOL_EVENT_THREAD_CANCEL: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_THREAD_CANCEL]++;
            break;
        }
        case ABT_TOOL_EVENT_THREAD_YIELD: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_THREAD_YIELD]++;

#if !ABTXI_PROF_USE_TIME_LOCAL
            ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
#else
            ABTXI_PROF_LOCAL_T cur_time_local = ABTXI_prof_get_time_local();
            p_xstream_info->d.time_last_run_local = cur_time_local;
            if (ABTXI_prof_unlikely(p_xstream_info->d.time_first_run_local ==
                                    ABTXI_PROF_LOCAL_T_ZERO)) {
                p_xstream_info->d.time_first_run_local = cur_time_local;
            }
#endif

            int depth =
                ABTXI_prof_xstream_info_get_depth_thread(p_xstream_info, event,
                                                         context, 0);
            if (depth < ABTXI_PROF_MAX_DEPTH) {
#if !ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_T xstream_time_last_run =
                    p_xstream_info->d.times_last_run[depth];
                if (ABTXI_prof_unlikely(xstream_time_last_run ==
                                        ABTXI_PROF_T_ZERO)) {
                    /* Adjustment */
                    xstream_time_last_run = p_global->start_prof_time;
                    p_xstream_info->d.times_last_run[depth] =
                        xstream_time_last_run;
                }
                p_xstream_info->d.times_elapsed[depth] +=
                    cur_time - xstream_time_last_run;
                p_xstream_info->d.times_last_run[depth] =
                    ABTXI_PROF_TIME_LAST_RUN_INVALID;
#else
                ABTXI_PROF_LOCAL_T xstream_time_last_run_local =
                    p_xstream_info->d.times_last_run_local[depth];
                if (ABTXI_prof_unlikely(xstream_time_last_run_local ==
                                        ABTXI_PROF_LOCAL_T_ZERO)) {
                    /* Adjustment */
                    xstream_time_last_run_local = cur_time_local;
                    p_xstream_info->d.times_last_run_local[depth] =
                        cur_time_local;
                }
                p_xstream_info->d.times_elapsed_local[depth] +=
                    cur_time_local - xstream_time_last_run_local;
                p_xstream_info->d.times_last_run_local[depth] =
                    ABTXI_PROF_TIME_LAST_RUN_LOCAL_INVALID;
#endif
            }

            if (prof_mode == ABTX_PROF_MODE_DETAILED) {
#if ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
#endif
                ABTXI_PROF_T thread_time_last_run =
                    p_thread_info->d.time_last_run;
                if (ABTXI_prof_unlikely(thread_time_last_run ==
                                        ABTXI_PROF_T_ZERO)) {
                    /* Adjustment */
                    thread_time_last_run = p_global->start_prof_time;
                    p_thread_info->d.time_last_run = thread_time_last_run;
                }
                p_thread_info->d.time_elapsed +=
                    cur_time - thread_time_last_run;
#if ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_LOCAL_T thread_time_last_run_local =
                    p_thread_info->d.time_last_run_local;
                if (ABTXI_prof_unlikely(thread_time_last_run_local ==
                                        ABTXI_PROF_LOCAL_T_ZERO)) {
                    /* Adjustment */
                    thread_time_last_run_local = cur_time_local;
                    p_thread_info->d.time_last_run_local = cur_time_local;
                }
                p_thread_info->d.time_elapsed_local +=
                    cur_time_local - thread_time_last_run_local;
#endif
                p_thread_info->d.num_yields++;
            }
            break;
        }
        case ABT_TOOL_EVENT_THREAD_SUSPEND: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_THREAD_SUSPEND]++;

#if !ABTXI_PROF_USE_TIME_LOCAL
            ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
#else
            ABTXI_PROF_LOCAL_T cur_time_local = ABTXI_prof_get_time_local();
            p_xstream_info->d.time_last_run_local = cur_time_local;
            if (ABTXI_prof_unlikely(p_xstream_info->d.time_first_run_local ==
                                    ABTXI_PROF_LOCAL_T_ZERO)) {
                p_xstream_info->d.time_first_run_local = cur_time_local;
            }
#endif

            int depth =
                ABTXI_prof_xstream_info_get_depth_thread(p_xstream_info, event,
                                                         context, 0);
            if (depth < ABTXI_PROF_MAX_DEPTH) {
#if !ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_T xstream_time_last_run =
                    p_xstream_info->d.times_last_run[depth];
                if (ABTXI_prof_unlikely(xstream_time_last_run ==
                                        ABTXI_PROF_T_ZERO)) {
                    /* Adjustment */
                    xstream_time_last_run = p_global->start_prof_time;
                    p_xstream_info->d.times_last_run[depth] =
                        xstream_time_last_run;
                }
                p_xstream_info->d.times_elapsed[depth] +=
                    cur_time - xstream_time_last_run;
                p_xstream_info->d.times_last_run[depth] =
                    ABTXI_PROF_TIME_LAST_RUN_INVALID;
#else
                ABTXI_PROF_LOCAL_T xstream_time_last_run_local =
                    p_xstream_info->d.times_last_run_local[depth];
                if (ABTXI_prof_unlikely(xstream_time_last_run_local ==
                                        ABTXI_PROF_LOCAL_T_ZERO)) {
                    /* Adjustment */
                    xstream_time_last_run_local = cur_time_local;
                    p_xstream_info->d.times_last_run_local[depth] =
                        cur_time_local;
                }
                p_xstream_info->d.times_elapsed_local[depth] +=
                    cur_time_local - xstream_time_last_run_local;
                p_xstream_info->d.times_last_run_local[depth] =
                    ABTXI_PROF_TIME_LAST_RUN_LOCAL_INVALID;
#endif
            }

            if (prof_mode == ABTX_PROF_MODE_DETAILED) {
#if ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
#endif
                ABTXI_PROF_T thread_time_last_run =
                    p_thread_info->d.time_last_run;
                if (ABTXI_prof_unlikely(thread_time_last_run ==
                                        ABTXI_PROF_T_ZERO)) {
                    /* Adjustment */
                    thread_time_last_run = p_global->start_prof_time;
                    p_thread_info->d.time_last_run = thread_time_last_run;
                }
                p_thread_info->d.time_elapsed +=
                    cur_time - thread_time_last_run;
#if ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_LOCAL_T thread_time_last_run_local =
                    p_thread_info->d.time_last_run_local;
                if (ABTXI_prof_unlikely(thread_time_last_run_local ==
                                        ABTXI_PROF_LOCAL_T_ZERO)) {
                    /* Adjustment */
                    thread_time_last_run_local = cur_time_local;
                    p_thread_info->d.time_last_run_local = cur_time_local;
                }
                p_thread_info->d.time_elapsed_local +=
                    cur_time_local - thread_time_last_run_local;
#endif
                p_thread_info->d.num_suspensions++;
            }
            break;
        }
        case ABT_TOOL_EVENT_THREAD_RESUME: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_THREAD_RESUME]++;
            break;
        }
    }
    /* Enter the clean phase. */
    ABTXI_prof_atomic_release_store_int(&p_xstream_info->tag,
                                        p_xstream_info->tag + 1);
}

static ABTXI_prof_always_inline void
ABTXI_prof_task_callback_impl(ABT_task task, ABT_xstream xstream,
                              uint64_t event, ABT_tool_context context,
                              void *user_arg, int prof_mode)
{
    ABTXI_prof_global *p_global = (ABTXI_prof_global *)user_arg;
    ABTXI_prof_xstream_info *p_xstream_info =
        ABTXI_prof_get_xstream_info(p_global, xstream);

    /* Enter the dirty phase.  This is needed to avoid the print function
     * reading premature values. */
    ABTXI_prof_atomic_release_store_int(&p_xstream_info->tag,
                                        p_xstream_info->tag + 1);

    ABTXI_prof_task_info *p_task_info;
    if (prof_mode == ABTX_PROF_MODE_DETAILED) {
        p_task_info = ABTXI_prof_get_task_info(p_global, p_xstream_info, task);
    }
    switch (event) {
        case ABT_TOOL_EVENT_TASK_CREATE: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_TASK_CREATE]++;

            if (prof_mode == ABTX_PROF_MODE_DETAILED) {
                ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
                p_task_info->d.time_created = cur_time;
            }
            break;
        }
        case ABT_TOOL_EVENT_TASK_JOIN: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_TASK_JOIN]++;
            break;
        }
        case ABT_TOOL_EVENT_TASK_FREE: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_TASK_FREE]++;

            if (prof_mode == ABTX_PROF_MODE_DETAILED) {
                ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
                /* task_info is no longer needed. */
                ABTXI_prof_release_task_info(p_xstream_info, p_task_info,
                                             cur_time);
            }
            break;
        }
        case ABT_TOOL_EVENT_TASK_REVIVE: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_TASK_REVIVE]++;

            if (prof_mode == ABTX_PROF_MODE_DETAILED) {
                p_task_info->d.num_revivals++;
            }
            break;
        }
        case ABT_TOOL_EVENT_TASK_RUN: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_TASK_RUN]++;

#if !ABTXI_PROF_USE_TIME_LOCAL
            ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
#else
            ABTXI_PROF_LOCAL_T cur_time_local = ABTXI_prof_get_time_local();
            p_xstream_info->d.time_last_run_local = cur_time_local;
            if (ABTXI_prof_unlikely(p_xstream_info->d.time_first_run_local ==
                                    ABTXI_PROF_LOCAL_T_ZERO)) {
                p_xstream_info->d.time_first_run_local = cur_time_local;
            }
#endif

            int depth =
                ABTXI_prof_xstream_info_get_depth_task(p_xstream_info, event,
                                                       context, 1);
            if (depth < ABTXI_PROF_MAX_DEPTH) {
#if !ABTXI_PROF_USE_TIME_LOCAL
                p_xstream_info->d.times_last_run[depth] = cur_time;
#else
                p_xstream_info->d.times_last_run_local[depth] = cur_time_local;
#endif
            }

            if (prof_mode == ABTX_PROF_MODE_DETAILED) {
#if ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
#endif
                if (p_task_info->d.time_first_run == ABTXI_PROF_T_ZERO) {
                    p_task_info->d.time_first_run = cur_time;
                }
                p_task_info->d.time_last_run = cur_time;
#if ABTXI_PROF_USE_TIME_LOCAL
                p_task_info->d.time_last_run_local = cur_time_local;
#endif
            }
            break;
        }
        case ABT_TOOL_EVENT_TASK_FINISH: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_TASK_FINISH]++;

#if !ABTXI_PROF_USE_TIME_LOCAL
            ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
#else
            ABTXI_PROF_LOCAL_T cur_time_local = ABTXI_prof_get_time_local();
            p_xstream_info->d.time_last_run_local = cur_time_local;
            if (ABTXI_prof_unlikely(p_xstream_info->d.time_first_run_local ==
                                    ABTXI_PROF_LOCAL_T_ZERO)) {
                p_xstream_info->d.time_first_run_local = cur_time_local;
            }
#endif

            int depth =
                ABTXI_prof_xstream_info_get_depth_task(p_xstream_info, event,
                                                       context, 0);
            if (depth < ABTXI_PROF_MAX_DEPTH) {
#if !ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_T xstream_time_last_run =
                    p_xstream_info->d.times_last_run[depth];
                if (ABTXI_prof_unlikely(xstream_time_last_run ==
                                        ABTXI_PROF_T_ZERO)) {
                    /* Adjustment */
                    xstream_time_last_run = p_global->start_prof_time;
                    p_xstream_info->d.times_last_run[depth] =
                        xstream_time_last_run;
                }
                p_xstream_info->d.times_elapsed[depth] +=
                    cur_time - xstream_time_last_run;
                p_xstream_info->d.times_last_run[depth] =
                    ABTXI_PROF_TIME_LAST_RUN_INVALID;
#else
                ABTXI_PROF_LOCAL_T xstream_time_last_run_local =
                    p_xstream_info->d.times_last_run_local[depth];
                if (ABTXI_prof_unlikely(xstream_time_last_run_local ==
                                        ABTXI_PROF_LOCAL_T_ZERO)) {
                    /* Adjustment */
                    xstream_time_last_run_local = cur_time_local;
                    p_xstream_info->d.times_last_run_local[depth] =
                        xstream_time_last_run_local;
                }
                p_xstream_info->d.times_elapsed_local[depth] +=
                    cur_time_local - xstream_time_last_run_local;
                p_xstream_info->d.times_last_run_local[depth] =
                    ABTXI_PROF_TIME_LAST_RUN_LOCAL_INVALID;
#endif
            }

            if (prof_mode == ABTX_PROF_MODE_DETAILED) {
#if ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_T cur_time = ABTXI_prof_get_time();
#endif
                ABTXI_PROF_T task_time_last_run = p_task_info->d.time_last_run;
                if (ABTXI_prof_unlikely(task_time_last_run ==
                                        ABTXI_PROF_T_ZERO)) {
                    /* Adjustment */
                    task_time_last_run = p_global->start_prof_time;
                    p_task_info->d.time_last_run = task_time_last_run;
                }
                p_task_info->d.time_elapsed += cur_time - task_time_last_run;
#if ABTXI_PROF_USE_TIME_LOCAL
                ABTXI_PROF_LOCAL_T task_time_last_run_local =
                    p_task_info->d.time_last_run_local;
                if (ABTXI_prof_unlikely(task_time_last_run_local ==
                                        ABTXI_PROF_LOCAL_T_ZERO)) {
                    /* Adjustment */
                    task_time_last_run_local = cur_time_local;
                    p_task_info->d.time_last_run_local =
                        task_time_last_run_local;
                }
                p_task_info->d.time_elapsed_local +=
                    cur_time_local - task_time_last_run_local;
#endif
                p_task_info->d.time_last_finish = cur_time;
            }
            break;
        }
        case ABT_TOOL_EVENT_TASK_CANCEL: {
            p_xstream_info->d.num_events[ABTXI_PROF_EVENT_TASK_CANCEL]++;
            break;
        }
    }
    /* Enter the clean phase. */
    ABTXI_prof_atomic_release_store_int(&p_xstream_info->tag,
                                        p_xstream_info->tag + 1);
}

static void ABTXI_prof_thread_callback_basic(ABT_thread thread,
                                             ABT_xstream xstream,
                                             uint64_t event,
                                             ABT_tool_context context,
                                             void *user_arg)
{
    ABTXI_prof_thread_callback_impl(thread, xstream, event, context, user_arg,
                                    ABTX_PROF_MODE_BASIC);
}

static void ABTXI_prof_thread_callback_detailed(ABT_thread thread,
                                                ABT_xstream xstream,
                                                uint64_t event,
                                                ABT_tool_context context,
                                                void *user_arg)
{
    ABTXI_prof_thread_callback_impl(thread, xstream, event, context, user_arg,
                                    ABTX_PROF_MODE_DETAILED);
}

static void ABTXI_prof_task_callback_basic(ABT_task task, ABT_xstream xstream,
                                           uint64_t event,
                                           ABT_tool_context context,
                                           void *user_arg)
{
    ABTXI_prof_task_callback_impl(task, xstream, event, context, user_arg,
                                  ABTX_PROF_MODE_BASIC);
}

static void ABTXI_prof_task_callback_detailed(ABT_task task,
                                              ABT_xstream xstream,
                                              uint64_t event,
                                              ABT_tool_context context,
                                              void *user_arg)
{
    ABTXI_prof_task_callback_impl(task, xstream, event, context, user_arg,
                                  ABTX_PROF_MODE_DETAILED);
}

static void ABTXI_prof_print_table_dsv(ABTXI_prof_data_table *p_table,
                                       const char *delimiter, FILE *stream)
{
    int col, row;
    for (col = 0; col < p_table->num_columns; col++) {
        fprintf(stream, "%s%s", delimiter, p_table->column_names[col]);
    }
    fprintf(stream, "\n");
    for (row = 0; row < p_table->num_rows; row++) {
        fprintf(stream, "%s", p_table->row_names[row]);
        for (col = 0; col < p_table->num_columns; col++) {
            double val = p_table->values[row * p_table->num_columns + col];
            if (val == (double)((long long int)val)) {
                fprintf(stream, "%s%lld", delimiter, (long long int)val);
            } else {
                fprintf(stream, "%s%.10f", delimiter, val);
            }
        }
        fprintf(stream, "\n");
    }
}

static int ABTXI_prof_ftos_fancy(char *str, double val, int digit)
{
    const int val_digit_min = -3; /* if it is -1: 1.00, 0.100, 1.00e-03, ... */
    const int val_digit_max = 5;  /* if it is 3: 12.3, 123, 1.23e+04, ... */
    const int val_accuracy = 3;   /* if it is 4: 1.235e+04 */
    if (digit <= -10) {
        /* Too small.  This is zero. */
        return sprintf(str, "0");
    } else if (digit < val_digit_min || val_digit_max < digit) {
        return sprintf(str, "%.*fe%+2d", val_accuracy - 1,
                       val / ABTXI_prof_pow(10.0, digit), digit);
    } else if (digit >= val_accuracy) {
        return sprintf(str, "%lld", (long long int)val);
    } else {
        if (val == (double)((long long int)val)) {
            return sprintf(str, "%lld", (long long int)val);
        } else {
            return sprintf(str, "%.*f", val_accuracy - digit - 1, val);
        }
    }
}

static void ABTXI_prof_print_table_fancy(ABTXI_prof_data_table *p_table,
                                         FILE *stream, int same_digit_from)
{
    int col, row;
    int *widths = (int *)calloc(p_table->num_columns + 1, sizeof(int));
    int *digits = (int *)malloc(p_table->num_rows * sizeof(int));
    char tmp[512];
    /* Calculate the digits. */
    for (row = 0; row < p_table->num_rows; row++) {
        digits[row] = -99;
        for (col = same_digit_from; col < p_table->num_columns; col++) {
            double val = p_table->values[row * p_table->num_columns + col];
            digits[row] = ABTXI_prof_max(ABTXI_prof_digit(val), digits[row]);
        }
    }

    /* The first column */
    for (row = 0; row < p_table->num_rows; row++) {
        int len = strlen(p_table->row_names[row]);
        widths[0] = ABTXI_prof_max(widths[0], len);
    }
    /* The other columns */
    for (col = 0; col < p_table->num_columns; col++) {
        widths[col + 1] = strlen(p_table->column_names[col]);
        for (row = 0; row < p_table->num_rows; row++) {
            double val = p_table->values[row * p_table->num_columns + col];
            int len;
            if (col >= same_digit_from) {
                len = ABTXI_prof_ftos_fancy(tmp, val, digits[row]);
            } else {
                len = ABTXI_prof_ftos_fancy(tmp, val, ABTXI_prof_digit(val));
            }
            widths[col + 1] = ABTXI_prof_max(widths[col + 1], len);
        }
    }

    fprintf(stream, "%*s", widths[0], "");
    for (col = 0; col < p_table->num_columns; col++) {
        fprintf(stream, "  %*s", widths[col + 1], p_table->column_names[col]);
    }
    fprintf(stream, "\n");
    for (row = 0; row < p_table->num_rows; row++) {
        fprintf(stream, "%*s", -widths[0], p_table->row_names[row]);
        for (col = 0; col < p_table->num_columns; col++) {
            double val = p_table->values[row * p_table->num_columns + col];
            if (col >= same_digit_from) {
                ABTXI_prof_ftos_fancy(tmp, val, digits[row]);
            } else {
                ABTXI_prof_ftos_fancy(tmp, val, ABTXI_prof_digit(val));
            }
            fprintf(stream, "  %*s", widths[col + 1], tmp);
        }
        fprintf(stream, "\n");
    }
    free(widths);
    free(digits);
}

static void
ABTXI_prof_merge_xstream_info(const ABTXI_prof_global *p_global,
                              const ABTXI_prof_xstream_info *p_xstream_info,
                              ABTXI_prof_xstream_data *p_out)
{
    int i;
    /* Copy thread_info and task_info and reduce it.  Note that unused _info
     * are initialized, so reducing them do not affect the results.  These
     * p_{thread/task}_all are managed so that so list traversal always
     * succeeds. */
    const ABTXI_prof_thread_info *p_thread_head =
        (const ABTXI_prof_thread_info *)ABTXI_prof_atomic_acquire_load_ptr(
            (void **)&p_xstream_info->p_thread_all);
    while (p_thread_head) {
        ABTXI_prof_merge_thread_data(p_out, &p_thread_head->d,
                                     ABTXI_PROF_T_ZERO);
        p_thread_head = p_thread_head->p_next_all;
    }
    const ABTXI_prof_task_info *p_task_head =
        (const ABTXI_prof_task_info *)ABTXI_prof_atomic_acquire_load_ptr(
            (void **)&p_xstream_info->p_task_all);
    while (p_task_head) {
        ABTXI_prof_merge_task_data(p_out, &p_task_head->d, ABTXI_PROF_T_ZERO);
        p_task_head = p_task_head->p_next_all;
    }

#if !ABTXI_PROF_USE_TIME_LOCAL
    ABTXI_PROF_T times_last_run[ABTXI_PROF_MAX_DEPTH];
    ABTXI_PROF_T times_elapsed[ABTXI_PROF_MAX_DEPTH];
    memcpy(times_last_run, &p_xstream_info->d.times_last_run,
           sizeof(ABTXI_PROF_T) * ABTXI_PROF_MAX_DEPTH);
    memcpy(times_elapsed, &p_xstream_info->d.times_elapsed,
           sizeof(ABTXI_PROF_T) * ABTXI_PROF_MAX_DEPTH);
    if (p_xstream_info->rank != -1) {
        ABTXI_PROF_T start_prof_time = p_global->start_prof_time;
        ABTXI_PROF_T stop_prof_time = p_global->stop_prof_time;
        /* Adjust times_elapsed.  We do not measure the execution time of
         * external threads, so adjustment is unnecessary. */
        if (p_out->cur_depth == 0) {
            /* The xstream did not really run anything, possibly because there
             * was no event on this xstream.  Let's think that at least the main
             * scheduler was active during that time. */
            times_elapsed[0] = stop_prof_time - start_prof_time;
        } else {
            for (i = 0; i < ABTXI_PROF_MAX_DEPTH; i++) {
                if (times_last_run[i] == ABTXI_PROF_T_ZERO &&
                    i < p_xstream_info->d.cur_depth) {
                    /* It means that this depth has not been executed, but the
                     * deeper level has been executed, implying this depth was
                     * active during this time frame. */
                    times_elapsed[i] += stop_prof_time - start_prof_time;
                } else if (times_last_run[i] != ABTXI_PROF_T_ZERO &&
                           times_last_run[i] !=
                               ABTXI_PROF_TIME_LAST_RUN_INVALID) {
                    /* It means that the last thread/task is still running on
                     * this xstream.  Let's add it. */
                    times_elapsed[i] += stop_prof_time - times_last_run[i];
                } else {
                    /* times_elapsed should be correct. */
                }
            }
        }
    }
    for (i = 0; i < ABTXI_PROF_MAX_DEPTH; i++)
        p_out->times_last_run[i] += times_last_run[i];
    for (i = 0; i < ABTXI_PROF_MAX_DEPTH; i++)
        p_out->times_elapsed[i] += times_elapsed[i];
#else
    /* No adjustment because this thread cannot access another ES's timers. */
    ABTXI_PROF_LOCAL_T times_last_run_local[ABTXI_PROF_MAX_DEPTH];
    ABTXI_PROF_LOCAL_T times_elapsed_local[ABTXI_PROF_MAX_DEPTH];
    memcpy(times_last_run_local, &p_xstream_info->d.times_last_run_local,
           sizeof(ABTXI_PROF_LOCAL_T) * ABTXI_PROF_MAX_DEPTH);
    memcpy(times_elapsed_local, &p_xstream_info->d.times_elapsed_local,
           sizeof(ABTXI_PROF_LOCAL_T) * ABTXI_PROF_MAX_DEPTH);
    if (p_xstream_info->rank != -1) {
        /* Adjust times_elapsed.  We do not measure the execution time of
         * external threads, so adjustment is unnecessary. */
        if (p_xstream_info->d.time_first_run_local == ABTXI_PROF_LOCAL_T_ZERO) {
            /* The xstream did not really run anything, possibly because there
             * was no event on this xstream.  Let's think that at least the main
             * scheduler was sleeping during that time. */
            times_elapsed_local[0] = 1.0e-9;
        } else {
            for (i = 0; i < ABTXI_PROF_MAX_DEPTH; i++) {
                if (times_last_run_local[i] == ABTXI_PROF_LOCAL_T_ZERO &&
                    i < p_xstream_info->d.cur_depth) {
                    /* It means that this depth has not been executed, but the
                     * deeper level has been executed, implying this depth was
                     * active during this time frame. */
                    times_elapsed_local[i] =
                        p_xstream_info->d.time_last_run_local -
                        p_xstream_info->d.time_first_run_local;
                } else if (times_last_run_local[i] != ABTXI_PROF_LOCAL_T_ZERO &&
                           times_last_run_local[i] !=
                               ABTXI_PROF_TIME_LAST_RUN_LOCAL_INVALID) {
                    /* It means that the last thread/task is still running on
                     * this xstream.  Let's add it. */
                    times_elapsed_local[i] +=
                        p_xstream_info->d.time_last_run_local -
                        times_last_run_local[i];
                } else {
                    /* times_elapsed should be correct. */
                }
            }
        }
    }
    for (i = 0; i < ABTXI_PROF_MAX_DEPTH; i++)
        p_out->times_last_run_local[i] += times_last_run_local[i];
    for (i = 0; i < ABTXI_PROF_MAX_DEPTH; i++)
        p_out->times_elapsed_local[i] += times_elapsed_local[i];
#endif

    for (i = 0; i < ABTXI_PROF_EVENT_END_; i++)
        p_out->num_events[i] += p_xstream_info->d.num_events[i];
    for (i = 0; i < ABTXI_PROF_WU_TIME_END_; i++)
        ABTXI_prof_wu_time_merge(&p_out->wu_times[i],
                                 &p_xstream_info->d.wu_times[i]);
    for (i = 0; i < ABTXI_PROF_WU_COUNT_END_; i++)
        ABTXI_prof_wu_count_merge(&p_out->wu_counts[i],
                                  &p_xstream_info->d.wu_counts[i]);
#if ABTXI_PROF_USE_TIME_LOCAL
    for (i = 0; i < ABTXI_PROF_WU_LOCAL_TIME_END_; i++)
        ABTXI_prof_wu_local_time_merge(&p_out->wu_local_times[i],
                                       &p_xstream_info->d.wu_local_times[i]);
#endif
}

static void ABTXI_prof_print_xstream_info(ABTXI_prof_global *p_global,
                                          FILE *stream, int print_mode)
{
    int i, j, rank;

    int num_ranks, len_ranks = 0, *ranks = NULL;
    ABTXI_prof_xstream_data *summaries;
    do {
        ABTXI_prof_spin_lock(&p_global->xstreams_lock);

        /* Get the number of execution streams, which should not be changed
         * while holding xstreams_lock. */
        num_ranks = 0;
        for (rank = 0; rank < p_global->len_p_xstreams; rank++) {
            if (p_global->p_xstreams[rank]) {
                if (num_ranks == len_ranks) {
                    len_ranks = ABTXI_prof_max(len_ranks * 2,
                                               ABTXI_PROF_DEFAULT_NUM_XSTREAMS);
                    ranks = (int *)realloc(ranks, sizeof(int) * len_ranks);
                }
                ranks[num_ranks] = rank;
                num_ranks++;
            }
        }
        int num_external_threads = 0, len_external_threads = 0;
        ABTXI_prof_xstream_info **external_threads = NULL;
        ABTXI_prof_xstream_info *p_xstream_info_head =
            p_global->p_xstream_info_head;
        while (p_xstream_info_head) {
            if (p_xstream_info_head->rank == -1) {
                /* Add new external threads. */
                if (num_external_threads == len_external_threads) {
                    len_external_threads =
                        ABTXI_prof_max(len_external_threads * 2, 16);
                    external_threads = (ABTXI_prof_xstream_info **)
                        realloc(external_threads,
                                sizeof(ABTXI_prof_xstream_info *) *
                                    len_external_threads);
                }
                external_threads[len_external_threads] = p_xstream_info_head;
                len_external_threads++;
            }
            p_xstream_info_head = p_xstream_info_head->p_next;
        }

        /* Record all the tags. */
        int *prev_tags =
            (int *)malloc(sizeof(int) * (num_ranks + num_external_threads));
        for (i = 0; i < num_ranks; i++)
            prev_tags[i] = ABTXI_prof_atomic_acquire_load_int(
                &p_global->p_xstreams[ranks[i]]->tag);
        for (i = 0; i < num_external_threads; i++)
            prev_tags[num_ranks + i] =
                ABTXI_prof_atomic_acquire_load_int(&external_threads[i]->tag);

        /* Read all the data.  The last one is for external threads. */
        summaries =
            (ABTXI_prof_xstream_data *)calloc(num_ranks + 1,
                                              sizeof(ABTXI_prof_xstream_data));
        for (i = 0; i < num_ranks; i++) {
            ABTXI_prof_merge_xstream_info(p_global,
                                          p_global->p_xstreams[ranks[i]],
                                          &summaries[i]);
        }
        for (i = 0; i < num_external_threads; i++) {
            ABTXI_prof_merge_xstream_info(p_global, external_threads[i],
                                          &summaries[num_ranks + 1]);
        }
        /* Check if all tags match the previous ones.  Otherwise, one of
         * callback handlers was called while reading xstream_info. */
        int fail = 0;
        for (i = 0; i < num_ranks; i++)
            fail |= prev_tags[i] != ABTXI_prof_atomic_acquire_load_int(
                                        &p_global->p_xstreams[ranks[i]]->tag);
        for (i = 0; i < num_external_threads; i++)
            fail |=
                prev_tags[num_ranks + i] !=
                ABTXI_prof_atomic_acquire_load_int(&external_threads[i]->tag);
        free(prev_tags);
        free(external_threads);
        ABTXI_prof_spin_unlock(&p_global->xstreams_lock);

        if (fail) {
            /* Failed.  Try again. */
            free(summaries);
            continue;
        }
        /* Succeeded. */
    } while (0);

    /* Reduce thread/task information. */
    ABTXI_prof_wu_time wu_times[ABTXI_PROF_WU_TIME_END_];
    memset(&wu_times, 0, sizeof(ABTXI_prof_wu_time) * ABTXI_PROF_WU_TIME_END_);
    ABTXI_prof_wu_count wu_counts[ABTXI_PROF_WU_COUNT_END_];
    memset(&wu_counts, 0,
           sizeof(ABTXI_prof_wu_count) * ABTXI_PROF_WU_COUNT_END_);
#if ABTXI_PROF_USE_TIME_LOCAL
    ABTXI_prof_wu_local_time wu_local_times[ABTXI_PROF_WU_LOCAL_TIME_END_];
    memset(&wu_local_times, 0,
           sizeof(ABTXI_prof_wu_local_time) * ABTXI_PROF_WU_LOCAL_TIME_END_);
#endif
    for (i = 0; i < num_ranks; i++) {
        for (j = 0; j < ABTXI_PROF_WU_TIME_END_; j++)
            ABTXI_prof_wu_time_merge(&wu_times[j], &summaries[i].wu_times[j]);
        for (j = 0; j < ABTXI_PROF_WU_COUNT_END_; j++)
            ABTXI_prof_wu_count_merge(&wu_counts[j],
                                      &summaries[i].wu_counts[j]);
#if ABTXI_PROF_USE_TIME_LOCAL
        for (j = 0; j < ABTXI_PROF_WU_LOCAL_TIME_END_; j++)
            ABTXI_prof_wu_local_time_merge(&wu_local_times[j],
                                           &summaries[i].wu_local_times[j]);
#endif
    }

    /* Print it. */
    if (print_mode & ABTX_PRINT_MODE_RAW) {
        {
            /* CSV or Fancy detailed (xstream_info). */
            ABTXI_prof_data_table table;
            ABTXI_prof_str_mem *p_str = ABTXI_prof_str_mem_alloc(4096);
            table.num_columns = num_ranks + 1;
            table.num_rows = ABTXI_PROF_EVENT_END_ + ABTXI_PROF_MAX_DEPTH;
            table.values = (double *)malloc(sizeof(double) * table.num_columns *
                                            table.num_rows);
            table.column_names =
                (const char **)malloc(sizeof(char *) * table.num_columns);
            table.row_names =
                (const char **)malloc(sizeof(char *) * table.num_rows);
            /* Set the column names. */
            for (i = 0; i < num_ranks; i++)
                table.column_names[i] =
                    ABTXI_prof_sprintf(p_str, 1024, "ES-%d", ranks[i]);
            table.column_names[num_ranks] = "Ext";
            /* Set the row names */
            for (i = 0; i < ABTXI_PROF_EVENT_END_; i++)
                table.row_names[i] =
                    ABTXI_prof_sprintf(p_str, 1024, "# of events of %s",
                                       ABTXI_get_prof_event_name(i));
#if !ABTXI_PROF_USE_TIME_LOCAL
            for (i = 0; i < ABTXI_PROF_MAX_DEPTH; i++)
                table.row_names[ABTXI_PROF_EVENT_END_ + i] =
                    ABTXI_prof_sprintf(p_str, 1024,
                                       "Level %d elapsed (approx.) [%s]", i,
                                       ABTXI_PROF_T_STRING);
#else
            for (i = 0; i < ABTXI_PROF_MAX_DEPTH; i++)
                table.row_names[ABTXI_PROF_EVENT_END_ + i] =
                    ABTXI_prof_sprintf(p_str, 1024, "Level %d elapsed [%s]", i,
                                       ABTXI_PROF_LOCAL_T_STRING);
#endif
            /* Set the data. */
            for (i = 0; i < num_ranks + 1; i++) {
                for (j = 0; j < ABTXI_PROF_EVENT_END_; j++) {
                    table.values[j * table.num_columns + i] =
                        summaries[i].num_events[j];
                }
                for (j = 0; j < ABTXI_PROF_MAX_DEPTH; j++) {
                    int row = j + ABTXI_PROF_EVENT_END_;
#if !ABTXI_PROF_USE_TIME_LOCAL
                    table.values[row * table.num_columns + i] =
                        summaries[i].times_elapsed[j];
#else
                    table.values[row * table.num_columns + i] =
                        summaries[i].times_elapsed_local[j];
#endif
                }
            }
            if (print_mode & ABTX_PRINT_MODE_CSV) {
                ABTXI_prof_print_table_dsv(&table, ", ", stream);
            } else {
                ABTXI_prof_print_table_fancy(&table, stream, 0);
            }
            free(table.values);
            free(table.row_names);
            free(table.column_names);
            ABTXI_prof_str_mem_free(p_str);
        }
        if (p_global->prof_mode == ABTX_PROF_MODE_DETAILED) {
            /* CSV or Fancy detailed (thread_info). */
            ABTXI_prof_data_table table;
            ABTXI_prof_str_mem *p_str = ABTXI_prof_str_mem_alloc(4096);
            table.num_columns = 4; /* Sum / Avg / Min / Max */
            table.num_rows = ABTXI_PROF_WU_TIME_END_ + ABTXI_PROF_WU_COUNT_END_;
#if ABTXI_PROF_USE_TIME_LOCAL
            table.num_rows += ABTXI_PROF_WU_LOCAL_TIME_END_;
#endif
            table.values =
                (double *)malloc(sizeof(double) * 4 * table.num_rows);
            table.column_names = (const char **)malloc(sizeof(char *) * 4);
            table.row_names =
                (const char **)malloc(sizeof(char *) * table.num_rows);
            /* Set the column names. */
            table.column_names[0] = "Sum";
            table.column_names[1] = "Avg";
            table.column_names[2] = "Min";
            table.column_names[3] = "Max";
            /* Set the row names */
            int row = 0;
            for (i = 0; i < ABTXI_PROF_WU_TIME_END_; i++, row++)
                table.row_names[row] = ABTXI_get_prof_wu_time_name(i);
#if ABTXI_PROF_USE_TIME_LOCAL
            for (i = 0; i < ABTXI_PROF_WU_LOCAL_TIME_END_; i++, row++)
                table.row_names[row] = ABTXI_get_prof_wu_local_time_name(i);
#endif
            for (i = 0; i < ABTXI_PROF_WU_COUNT_END_; i++, row++)
                table.row_names[row] =
                    ABTXI_prof_sprintf(p_str, 1024, "# of events of %s",
                                       ABTXI_get_prof_wu_count_name(i));
            /* Set the data. */
            row = 0;
            for (i = 0; i < ABTXI_PROF_WU_TIME_END_; i++, row++) {
                table.values[row * 4 + 0] = wu_times[i].sum;
                table.values[row * 4 + 1] =
                    ABTXI_prof_div_s(wu_times[i].sum, wu_times[i].cnt);
                table.values[row * 4 + 2] = wu_times[i].min_val;
                table.values[row * 4 + 3] = wu_times[i].max_val;
            }
#if ABTXI_PROF_USE_TIME_LOCAL
            for (i = 0; i < ABTXI_PROF_WU_LOCAL_TIME_END_; i++, row++) {
                table.values[row * 4 + 0] = wu_local_times[i].sum;
                table.values[row * 4 + 1] =
                    ABTXI_prof_div_s(wu_local_times[i].sum,
                                     wu_local_times[i].cnt);
                table.values[row * 4 + 2] = wu_local_times[i].min_val;
                table.values[row * 4 + 3] = wu_local_times[i].max_val;
            }
#endif
            for (i = 0; i < ABTXI_PROF_WU_COUNT_END_; i++, row++) {
                table.values[row * 4 + 0] = wu_counts[i].sum;
                table.values[row * 4 + 1] =
                    ABTXI_prof_div_s((double)wu_counts[i].sum,
                                     wu_counts[i].cnt);
                table.values[row * 4 + 2] = wu_counts[i].min_val;
                table.values[row * 4 + 3] = wu_counts[i].max_val;
            }
            if (print_mode & ABTX_PRINT_MODE_CSV) {
                ABTXI_prof_print_table_dsv(&table, ", ", stream);
            } else {
                ABTXI_prof_print_table_fancy(&table, stream, 1);
            }
            free(table.values);
            free(table.row_names);
            free(table.column_names);
            ABTXI_prof_str_mem_free(p_str);
        }
    } else if (print_mode & ABTX_PRINT_MODE_SUMMARY) {
        const double to_sec = p_global->to_sec;
        const double elapsed_time =
            (p_global->stop_prof_time - p_global->start_prof_time) * to_sec;
        {
            int row = -1;
            ABTXI_prof_data_table table;
            ABTXI_prof_str_mem *p_str = ABTXI_prof_str_mem_alloc(4096);
            table.num_columns = num_ranks + 1; /* Average / (Ranks ...) */
            table.num_rows = 5;
            table.values = (double *)calloc(table.num_columns * table.num_rows,
                                            sizeof(double));
            table.column_names =
                (const char **)malloc(sizeof(char *) * table.num_columns);
            table.row_names =
                (const char **)malloc(sizeof(char *) * table.num_rows);
            /* Set the column names. */
            table.column_names[0] = "Average";
            for (i = 0; i < num_ranks; i++)
                table.column_names[i + 1] =
                    ABTXI_prof_sprintf(p_str, 1024, "ES-%d", ranks[i]);

            /* External threads cannot execute ULTs, so all the following values
             * should be 0. */
            table.row_names[++row] = "Approx. ULT/tasklet granularity [s]";
            for (i = 0; i < num_ranks; i++) {
                /* Assume that no ULT/tasklet revived and no stackable
                 * scheduler. */
                int num_finishes =
                    summaries[i].num_events[ABTXI_PROF_EVENT_THREAD_FINISH] +
                    summaries[i].num_events[ABTXI_PROF_EVENT_TASK_FINISH];
#if !ABTXI_PROF_USE_TIME_LOCAL
                double granularity =
                    ABTXI_prof_div_s(summaries[i].times_elapsed[1] * to_sec,
                                     num_finishes);
#else
                double granularity =
                    ABTXI_prof_div_s(summaries[i].times_elapsed_local[1],
                                     num_finishes);
#endif
                table.values[row * table.num_columns] +=
                    granularity / num_ranks;
                table.values[row * table.num_columns + i + 1] = granularity;
            }
            table.row_names[++row] = "Approx. ULT/tasklet throughput [/s]";
            for (i = 0; i < num_ranks; i++) {
                int num_finishes =
                    summaries[i].num_events[ABTXI_PROF_EVENT_THREAD_FINISH] +
                    summaries[i].num_events[ABTXI_PROF_EVENT_TASK_FINISH];
                table.values[row * table.num_columns] +=
                    ABTXI_prof_div_s(num_finishes, elapsed_time) / num_ranks;
                table.values[row * table.num_columns + i + 1] =
                    ABTXI_prof_div_s(num_finishes, elapsed_time);
            }
#if !ABTXI_PROF_USE_TIME_LOCAL
            table.row_names[++row] = "Approx. non-main scheduling ratio [%]";
#else
            table.row_names[++row] = "Non-main scheduling ratio [%]";
#endif
            for (i = 0; i < num_ranks; i++) {
#if !ABTXI_PROF_USE_TIME_LOCAL
                double ratio = summaries[i].times_elapsed[1] * to_sec /
                               elapsed_time * 100.0;
#else
                double ratio = summaries[i].times_elapsed_local[1] /
                               summaries[i].times_elapsed_local[0] * 100.0;
#endif
                table.values[row * table.num_columns] += ratio / num_ranks;
                table.values[row * table.num_columns + i + 1] = ratio;
            }
            table.row_names[++row] = "# of events of ULT/yield [/s]";
            for (i = 0; i < num_ranks; i++) {
                double nyields =
                    summaries[i].num_events[ABTXI_PROF_EVENT_THREAD_YIELD] /
                    elapsed_time;
                table.values[row * table.num_columns] += nyields / num_ranks;
                table.values[row * table.num_columns + i + 1] = nyields;
            }
            table.row_names[++row] = "# of events of ULT/suspend [/s]";
            for (i = 0; i < num_ranks; i++) {
                double ncreates =
                    summaries[i].num_events[ABTXI_PROF_EVENT_THREAD_SUSPEND] /
                    elapsed_time;
                table.values[row * table.num_columns] += ncreates / num_ranks;
                table.values[row * table.num_columns + i + 1] = ncreates;
            }
            assert(row == table.num_rows - 1);
            if (print_mode & ABTX_PRINT_MODE_CSV) {
                ABTXI_prof_print_table_dsv(&table, ", ", stream);
            } else {
                ABTXI_prof_print_table_fancy(&table, stream, 0);
            }
            free(table.values);
            free(table.row_names);
            free(table.column_names);

            row = -1;
            table.num_columns =
                num_ranks + 2; /* Sum / (Ranks ...) / External */
            table.num_rows = 2;
            table.values = (double *)calloc(table.num_columns * table.num_rows,
                                            sizeof(double));
            table.column_names =
                (const char **)malloc(sizeof(char *) * table.num_columns);
            table.row_names =
                (const char **)malloc(sizeof(char *) * table.num_rows);
            /* Set the column names. */
            table.column_names[0] = "Sum";
            for (i = 0; i < num_ranks; i++)
                table.column_names[i + 1] =
                    ABTXI_prof_sprintf(p_str, 1024, "ES-%d", ranks[i]);
            table.column_names[num_ranks + 1] = "Ext";

            /* Update the column names. */
            table.row_names[++row] = "# of created ULTs/tasklets";
            for (i = 0; i < num_ranks + 1; i++) {
                double ncreates =
                    (summaries[i].num_events[ABTXI_PROF_EVENT_THREAD_CREATE] +
                     summaries[i].num_events[ABTXI_PROF_EVENT_TASK_CREATE]);
                table.values[row * table.num_columns] += ncreates;
                table.values[row * table.num_columns + i + 1] = ncreates;
            }
            table.row_names[++row] = "# of created ULTs/tasklets [/s]";
            for (i = 0; i < num_ranks + 2; i++) {
                double ncreates_throughput =
                    table.values[(row - 1) * table.num_columns + i] /
                    elapsed_time;
                table.values[row * table.num_columns + i] = ncreates_throughput;
            }
            assert(row == table.num_rows - 1);
            if (print_mode & ABTX_PRINT_MODE_CSV) {
                ABTXI_prof_print_table_dsv(&table, ", ", stream);
            } else {
                ABTXI_prof_print_table_fancy(&table, stream, 1);
            }
            free(table.values);
            free(table.row_names);
            free(table.column_names);
            ABTXI_prof_str_mem_free(p_str);
        }
        if (p_global->prof_mode == ABTX_PROF_MODE_DETAILED) {
            int row = -1;
            /* CSV or Fancy summary (thread_info). */
            ABTXI_prof_data_table table;
            ABTXI_prof_str_mem *p_str = ABTXI_prof_str_mem_alloc(4096);
            table.num_columns = 3; /* Avg / Min / Max */
            table.num_rows = 8;
#if ABTXI_PROF_USE_TIME_LOCAL
            table.num_rows += 1;
#endif
            table.values = (double *)malloc(sizeof(double) * table.num_columns *
                                            table.num_rows);
            table.column_names =
                (const char **)malloc(sizeof(char *) * table.num_columns);
            table.row_names =
                (const char **)malloc(sizeof(char *) * table.num_rows);
            /* Set the column names. */
            table.column_names[0] = "Avg";
            table.column_names[1] = "Min";
            table.column_names[2] = "Max";
            /* Set the row names */
            table.row_names[++row] = "ULT/tasklet execution time [s]";
            ABTXI_prof_wu_time t_elapsed;
            memcpy(&t_elapsed, &wu_times[ABTXI_PROF_WU_TIME_THREAD_ELAPSED],
                   sizeof(ABTXI_prof_wu_time));
            ABTXI_prof_wu_time_merge(&t_elapsed,
                                     &wu_times
                                         [ABTXI_PROF_WU_TIME_TASK_ELAPSED]);
            table.values[row * 3 + 0] =
                ABTXI_prof_div_s(t_elapsed.sum, t_elapsed.cnt) * to_sec;
            table.values[row * 3 + 1] = t_elapsed.min_val * to_sec;
            table.values[row * 3 + 2] = t_elapsed.max_val * to_sec;
#if ABTXI_PROF_USE_TIME_LOCAL
            table.row_names[++row] = "ULT/tasklet granularity (execution time "
                                     "- ES unscheduled time) [s]";
            ABTXI_prof_wu_local_time t_elapsed_local;
            memcpy(&t_elapsed_local,
                   &wu_local_times[ABTXI_PROF_WU_LOCAL_TIME_THREAD_ELAPSED],
                   sizeof(ABTXI_prof_wu_local_time));
            ABTXI_prof_wu_local_time_merge(
                &t_elapsed_local,
                &wu_local_times[ABTXI_PROF_WU_LOCAL_TIME_TASK_ELAPSED]);
            table.values[row * 3 + 0] =
                ABTXI_prof_div_s(t_elapsed_local.sum, t_elapsed_local.cnt);
            table.values[row * 3 + 1] = t_elapsed_local.min_val;
            table.values[row * 3 + 2] = t_elapsed_local.max_val;
#endif
            table.row_names[++row] = "# of yield events per ULT";
            table.values[row * 3 + 0] =
                ABTXI_prof_div_s((double)wu_counts
                                     [ABTXI_PROF_WU_COUNT_THREAD_NUM_YIELDS]
                                         .sum,
                                 wu_counts
                                     [ABTXI_PROF_WU_COUNT_THREAD_NUM_YIELDS]
                                         .cnt);
            table.values[row * 3 + 1] =
                wu_counts[ABTXI_PROF_WU_COUNT_THREAD_NUM_YIELDS].min_val;
            table.values[row * 3 + 2] =
                wu_counts[ABTXI_PROF_WU_COUNT_THREAD_NUM_YIELDS].max_val;
            table.row_names[++row] = "# of suspend events per ULT";
            table.values[row * 3 + 0] = ABTXI_prof_div_s(
                (double)wu_counts[ABTXI_PROF_WU_COUNT_THREAD_NUM_SUSPENSIONS]
                    .sum,
                wu_counts[ABTXI_PROF_WU_COUNT_THREAD_NUM_SUSPENSIONS].cnt);
            table.values[row * 3 + 1] =
                wu_counts[ABTXI_PROF_WU_COUNT_THREAD_NUM_SUSPENSIONS].min_val;
            table.values[row * 3 + 2] =
                wu_counts[ABTXI_PROF_WU_COUNT_THREAD_NUM_SUSPENSIONS].max_val;
            table.row_names[++row] = "# of execution stream changes per ULT";
            table.values[row * 3 + 0] = ABTXI_prof_div_s(
                (double)
                    wu_counts[ABTXI_PROF_WU_COUNT_THREAD_NUM_XSTREAM_CHANGES]
                        .sum,
                wu_counts[ABTXI_PROF_WU_COUNT_THREAD_NUM_XSTREAM_CHANGES].cnt);
            table.values[row * 3 + 1] =
                wu_counts[ABTXI_PROF_WU_COUNT_THREAD_NUM_XSTREAM_CHANGES]
                    .min_val;
            table.values[row * 3 + 2] =
                wu_counts[ABTXI_PROF_WU_COUNT_THREAD_NUM_XSTREAM_CHANGES]
                    .max_val;
            table.row_names[++row] = "Active time per ULT/tasklet (last finish "
                                     "time - first execution time) [s]";
            ABTXI_prof_wu_time t_active;
            memcpy(&t_active,
                   &wu_times[ABTXI_PROF_WU_TIME_THREAD_FIRST_RUN_LAST_FINISH],
                   sizeof(ABTXI_prof_wu_time));
            ABTXI_prof_wu_time_merge(
                &t_active,
                &wu_times[ABTXI_PROF_WU_TIME_TASK_FIRST_RUN_LAST_FINISH]);
            table.values[row * 3 + 0] =
                ABTXI_prof_div_s(t_active.sum, t_active.cnt) * to_sec;
            table.values[row * 3 + 1] = t_active.min_val * to_sec;
            table.values[row * 3 + 2] = t_active.max_val * to_sec;
            table.row_names[++row] = "Execution delay per ULT/tasklet (first "
                                     "execution time - creation time) [s]";
            ABTXI_prof_wu_time t_delay;
            memcpy(&t_delay,
                   &wu_times[ABTXI_PROF_WU_TIME_THREAD_CREATE_FIRST_RUN],
                   sizeof(ABTXI_prof_wu_time));
            ABTXI_prof_wu_time_merge(
                &t_delay, &wu_times[ABTXI_PROF_WU_TIME_TASK_CREATE_FIRST_RUN]);
            table.values[row * 3 + 0] =
                ABTXI_prof_div_s(t_delay.sum, t_delay.cnt) * to_sec;
            table.values[row * 3 + 1] = t_delay.min_val * to_sec;
            table.values[row * 3 + 2] = t_delay.max_val * to_sec;
            table.row_names[++row] = "Completion time per ULT/tasklet (last "
                                     "finish time - creation time) [s]";
            ABTXI_prof_wu_time t_comp;
            memcpy(&t_comp,
                   &wu_times[ABTXI_PROF_WU_TIME_THREAD_CREATE_LAST_FINISH],
                   sizeof(ABTXI_prof_wu_time));
            ABTXI_prof_wu_time_merge(
                &t_comp, &wu_times[ABTXI_PROF_WU_TIME_TASK_CREATE_LAST_FINISH]);
            table.values[row * 3 + 0] =
                ABTXI_prof_div_s(t_comp.sum, t_comp.cnt) * to_sec;
            table.values[row * 3 + 1] = t_comp.min_val * to_sec;
            table.values[row * 3 + 2] = t_comp.max_val * to_sec;
            table.row_names[++row] =
                "Lifetime per ULT/tasklet (free time - create time) [s]";
            ABTXI_prof_wu_time t_life;
            memcpy(&t_life, &wu_times[ABTXI_PROF_WU_TIME_THREAD_CREATE_FREE],
                   sizeof(ABTXI_prof_wu_time));
            ABTXI_prof_wu_time_merge(&t_life,
                                     &wu_times
                                         [ABTXI_PROF_WU_TIME_TASK_CREATE_FREE]);
            table.values[row * 3 + 0] =
                ABTXI_prof_div_s(t_life.sum, t_life.cnt) * to_sec;
            table.values[row * 3 + 1] = t_life.min_val * to_sec;
            table.values[row * 3 + 2] = t_life.max_val * to_sec;
            if (print_mode & ABTX_PRINT_MODE_CSV) {
                ABTXI_prof_print_table_dsv(&table, ", ", stream);
            } else {
                ABTXI_prof_print_table_fancy(&table, stream, 9999);
            }
            free(table.values);
            free(table.row_names);
            free(table.column_names);
            ABTXI_prof_str_mem_free(p_str);
        }
    }

    free(summaries);
    free(ranks);
}

static inline int ABTX_prof_init(ABTX_prof_context *p_new_context)
{
    int ret;
    ABT_bool tool_enabled;
    ABT_info_query_config(ABT_INFO_QUERY_KIND_ENABLED_TOOL, &tool_enabled);
    if (tool_enabled != ABT_TRUE) {
        fprintf(stderr, "[ABTX_prof_init] The tool feature is disabled: "
                        "configure Argobots with --enable-tool.\n");
        return ABT_ERR_OTHER;
    }
    if (p_new_context == NULL) {
        fprintf(stderr, "[ABTX_prof_init] p_new_context may not be NULL.\n");
        return ABT_ERR_OTHER;
    }
    ret = ABT_initialized();
    if (ret != ABT_SUCCESS) {
        fprintf(stderr, "[ABTX_prof_init] Argobots is not initialized.  Call "
                        "ABTX_prof_init() after ABT_init(). \n");
        return ABT_ERR_UNINITIALIZED;
    }
    /* Create and allocate basic data. */
    ABTXI_prof_global *p_global =
        (ABTXI_prof_global *)calloc(1, sizeof(ABTXI_prof_global));
    p_global->state = ABTXI_PROF_GLOBAL_STATE_CLEAN;
    /* Allocate a Pthread key. */
    ret = pthread_key_create(&p_global->xstream_info_key, NULL);
    if (ret != 0) {
        fprintf(stderr, "[ABTX_prof_init] pthread_key_create() failed.  Too "
                        "many Pthread keys?\n");
        free(p_global);
        return ABT_ERR_OTHER;
    }
    /* Allocate a work unit key. */
    ABT_key_create(NULL, &p_global->prof_key);
    /* Initialize a spinlock. */
    ABTXI_prof_spin_init(&p_global->xstreams_lock);
    p_global->to_sec = ABTXI_prof_get_time_to_sec();
    *(ABTXI_prof_global **)p_new_context = p_global;
    return ABT_SUCCESS;
}

static inline int ABTX_prof_start(ABTX_prof_context context, int prof_mode)
{
    if (context == NULL) {
        fprintf(stderr, "[ABTX_prof_start] context may not be NULL.\n");
        return ABT_ERR_OTHER;
    }
    ABTXI_prof_global *p_global = (ABTXI_prof_global *)context;
    if (p_global->state == ABTXI_PROF_GLOBAL_STATE_RUNNING) {
        fprintf(stderr, "[ABTX_prof_start] ABTX_prof_start() is called twice "
                        "before ABTX_prof_stop().\n");
        return ABT_ERR_OTHER;
    }
    if (p_global->state == ABTXI_PROF_GLOBAL_STATE_STOPPED) {
        ABTXI_prof_xstream_info *p_cur = p_global->p_xstream_info_head;
        while (p_cur) {
            ABTXI_prof_xstream_info *p_next = p_cur->p_next;
            ABTXI_prof_reset_xstream_info(p_cur);
            p_cur = p_next;
        }
    }
    p_global->state = ABTXI_PROF_GLOBAL_STATE_RUNNING;
    p_global->prof_mode = prof_mode;
    p_global->start_prof_time = ABTXI_prof_get_time();
    if (prof_mode == ABTX_PROF_MODE_BASIC) {
        ABT_tool_register_thread_callback(ABTXI_prof_thread_callback_basic,
                                          ABT_TOOL_EVENT_THREAD_ALL, p_global);
        ABT_tool_register_task_callback(ABTXI_prof_task_callback_basic,
                                        ABT_TOOL_EVENT_TASK_ALL, p_global);
    } else {
        ABT_tool_register_thread_callback(ABTXI_prof_thread_callback_detailed,
                                          ABT_TOOL_EVENT_THREAD_ALL, p_global);
        ABT_tool_register_task_callback(ABTXI_prof_task_callback_detailed,
                                        ABT_TOOL_EVENT_THREAD_ALL, p_global);
    }
    return ABT_SUCCESS;
}

static inline int ABTX_prof_stop(ABTX_prof_context context)
{
    if (context == NULL) {
        fprintf(stderr, "[ABTX_prof_stop] context may not be NULL.\n");
        return ABT_ERR_OTHER;
    }
    ABTXI_prof_global *p_global = (ABTXI_prof_global *)context;
    if (p_global->state != ABTXI_PROF_GLOBAL_STATE_RUNNING) {
        fprintf(stderr,
                "[ABTX_prof_stop] ABTX_prof_start() has not been called.\n");
        return ABT_ERR_OTHER;
    }
    ABT_tool_register_thread_callback(NULL, ABT_TOOL_EVENT_THREAD_NONE, NULL);
    ABT_tool_register_task_callback(NULL, ABT_TOOL_EVENT_TASK_NONE, NULL);
    p_global->stop_prof_time = ABTXI_prof_get_time();
    p_global->state = ABTXI_PROF_GLOBAL_STATE_STOPPED;
    return ABT_SUCCESS;
}

static inline int ABTX_prof_print(ABTX_prof_context context, FILE *stream,
                                  int print_mode)
{
    if (context == NULL) {
        fprintf(stderr, "[ABTX_prof_print] context may not be NULL.\n");
        return ABT_ERR_OTHER;
    }
    ABTXI_prof_global *p_global = (ABTXI_prof_global *)context;
    if (p_global->state != ABTXI_PROF_GLOBAL_STATE_STOPPED) {
        fprintf(stderr,
                "[ABTX_prof_print] ABTX_prof_stop() has not been called.\n");
        return ABT_ERR_OTHER;
    }
    ABTXI_prof_print_xstream_info(p_global, stream, print_mode);
    return ABT_SUCCESS;
}

static inline int ABTX_prof_clean(ABTX_prof_context context)
{
    if (context == NULL) {
        fprintf(stderr, "[ABTX_prof_clean] context may not be NULL.\n");
        return ABT_ERR_OTHER;
    }
    ABTXI_prof_global *p_global = (ABTXI_prof_global *)context;
    if (p_global->state != ABTXI_PROF_GLOBAL_STATE_STOPPED) {
        fprintf(stderr,
                "[ABTX_prof_clean] ABTX_prof_stop() has not been called.\n");
        return ABT_ERR_OTHER;
    }
    ABTXI_prof_xstream_info *p_cur = p_global->p_xstream_info_head;
    while (p_cur) {
        ABTXI_prof_xstream_info *p_next = p_cur->p_next;
        ABTXI_prof_reset_xstream_info(p_cur);
        p_cur = p_next;
    }
    p_global->state = ABTXI_PROF_GLOBAL_STATE_CLEAN;
    return ABT_SUCCESS;
}

static inline int ABTX_prof_finalize(ABTX_prof_context context)
{
    int ret;
    if (context == NULL) {
        fprintf(stderr, "[ABTX_prof_finalize] context may not be NULL.\n");
        return ABT_ERR_OTHER;
    }
    ret = ABT_initialized();
    if (ret != ABT_SUCCESS) {
        fprintf(stderr, "[ABTX_prof_finalize] Argobots is not initialized.  "
                        "Call ABTX_prof_finalize() before ABT_finalize().\n");
        return ABT_ERR_UNINITIALIZED;
    }
    ABTXI_prof_global *p_global = (ABTXI_prof_global *)context;
    if (p_global->state == ABTXI_PROF_GLOBAL_STATE_RUNNING) {
        fprintf(stderr,
                "[ABTX_prof_finalize] The profiler is still running.\n");
        return ABT_ERR_OTHER;
    }
    /* Free xstreams. */
    ABTXI_prof_xstream_info *p_cur = p_global->p_xstream_info_head;
    while (p_cur) {
        ABTXI_prof_xstream_info *p_next = p_cur->p_next;
        ABTXI_prof_destroy_xstream_info(p_cur);
        free(p_cur);
        p_cur = p_next;
    }
    /* Free mem_p_xstreams. */
    void *mem_p_xstreams = p_global->mem_p_xstreams;
    while (mem_p_xstreams) {
        void *p_next = *(void **)mem_p_xstreams;
        free(mem_p_xstreams);
        mem_p_xstreams = p_next;
    }
    /* Delete a work unit key. */
    ABT_key_free(&p_global->prof_key);
    /* Delete a Pthread key. */
    pthread_key_delete(p_global->xstream_info_key);
    /* Delete a spinlock. */
    ABTXI_prof_spin_destroy(&p_global->xstreams_lock);
    free(p_global);
    return ABT_SUCCESS;
}

#endif /* ABTX_PROF_H_INCLUDED */
