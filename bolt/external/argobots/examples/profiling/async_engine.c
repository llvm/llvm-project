/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * This example implements a small asynchronous event/task/request engine.
 * The primary execution stream creates several "operations" (e.g., network or
 * I/O operations in real use cases) associated with ULTs.  These ULTs are
 * executed by multiple secondary execution streams.  Users can change the
 * following parameters:
 *
 * -e [NUM_ESS]   the number of secondary execution streams + 1
 * -n [NUM_OPS]   the total number of operations issued by the primary execution
 *                stream
 * -t [TIME]      total issuing time.  Throughput will be at most NUM_OPS / TIME
 * -s [SIZE]      computation size of each operation
 * -m [WAIT]      use ABT_POOL_FIFO_WAIT or not.
 * -p [PROF_MODE] 0: disabled, 1: basic, 2: detailed.
 *
 * The input parameters affect the following performance numbers.
 *
 * - Thread granularity
 *   This thread granularity is determined by SIZE; the granularity should be
 *   almost proportional to the value of SIZE.
 *
 * - Approx. ULT/tasklet throughput
 *   If all the operations are executed without stagnation, "the average
 *   throughput" * NUM_ESS should be the same as NUM_OPS / TIME.  If the
 *   operation issuing speed exceeds the operation processing speed, "the
 *   average throughput" * NUM_ESS should be less than NUM_OPS / TIME.
 *
 * - Non-main scheduling ratio
 *   If there is always plenty of operations, all the secondary execution
 *   streams are always busy, making this ratio high.  If NUM_OPS / TIME is
 *   small, however, most of the secondary execution streams are idle.  In such
 *   a case, "-m 1" (WAIT=1) would be helpful: this setting can sleep such
 *   secondary execution streams and thus increases this ratio.
 *   Note that this requires #define ABTX_PROF_ASSUME_SCHED_ALWAYS_ACTIVE=0 for
 *   accurate profiling.
 *
 * If you enable a detailed profiling mode (-p 2), you can get the following
 * information.
 *
 * - Execution delay per ULT/tasklet
 *   This shows how long it takes to schedule a thread for the first time after
 *   creating that thread.  This number is similar to "latency" but it focuses
 *   on the first scheduling timing.  WAIT=1 can increase this delay since it
 *   may suspend underlying execution streams in ABT_pool_wait().
 *
 * - Completion time per ULT/tasklet
 *   This shows how long it takes to complete (not "free") a thread after
 *   creating that thread.  This number is similar to "latency" but it focuses
 *   on completion.  WAIT=1 can increase this value since it may suspend
 *   underlying execution streams in ABT_pool_wait().
 *
 * This example also shows when to use ABTX_PROF_ASSUME_SCHED_ALWAYS_ACTIVE.  If
 * developers want to know "real execution time" of backend execution streams,
 * please set zero.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <stdarg.h>
#include <abt.h>

/* This assumption does not hold in this program since the schedulers might
 * sleep in ABT_pool_pop_timedwait(). */
#define ABTX_PROF_ASSUME_SCHED_ALWAYS_ACTIVE 0
#include "abtx_prof.h"

#define DEFAULT_SIZE 1024
#define DEFAULT_NUM_XSTREAMS 3 /* Must be larger than 1. */
#define DEFAULT_N 1024
#define DEFAULT_WAIT 0
#define DEFAULT_DURATION 5.0

#define THREAD_POOL_SIZE 256  /* The maximum number of ops in the pool. */
#define POOL_POP_WAIT_SEC 0.1 /* [s] */

int g_wait_mode;

void compute(int n)
{
    volatile double value = n + 1;
    int i;
    for (i = 0; i < n * 100; i++) {
        value = value * 0.5 + 1.0;
    }
    if (value == 0.0) {
        /* Unreachable.  This branch is to avoid compiler optimizations. */
        printf("compute() is broken\n");
    }
}

typedef struct {
    ABT_thread thread;
    int size;
} operation_arg_t;

void operation(void *arg)
{
    int size = ((operation_arg_t *)arg)->size;
    compute(size);
}

/* Scheduler */

int sched_init(ABT_sched sched, ABT_sched_config config)
{
    return ABT_SUCCESS;
}

void sched_run(ABT_sched sched)
{
    uint32_t work_count = 0;
    ABT_pool op_pool;
    ABT_sched_get_pools(sched, 1, 0, &op_pool);

    while (1) {
        ABT_unit unit;
        if (g_wait_mode == 0) {
            ABT_pool_pop(op_pool, &unit);
        } else {
            ABT_pool_pop_wait(op_pool, &unit, POOL_POP_WAIT_SEC);
        }
        if (unit != ABT_UNIT_NULL) {
            ABT_xstream_run_unit(unit, op_pool);
        }
        if (++work_count >= 128 || (g_wait_mode && unit == ABT_UNIT_NULL)) {
            ABT_bool stop;
            ABT_sched_has_to_stop(sched, &stop);
            if (stop == ABT_TRUE)
                break;
            work_count = 0;
            ABT_xstream_check_events(sched);
        }
    }
}

int sched_free(ABT_sched sched)
{
    return ABT_SUCCESS;
}

int main(int argc, char *argv[])
{
    int i;
    /* Read arguments. */
    int size = DEFAULT_SIZE;
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    int n = DEFAULT_N;
    double duration = DEFAULT_DURATION;
    g_wait_mode = DEFAULT_WAIT;
    int prof_mode = 1;
    while (1) {
        int opt = getopt(argc, argv, "he:n:t:s:w:p:");
        if (opt == -1)
            break;
        switch (opt) {
            case 'e':
                num_xstreams = atoi(optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 't':
                duration = atof(optarg);
                break;
            case 's':
                size = atoi(optarg);
                break;
            case 'w':
                g_wait_mode = atoi(optarg);
                break;
            case 'p':
                prof_mode = atoi(optarg);
                break;
            case 'h':
            default:
                printf(
                    "Usage: ./async_engine [-e NUM_XSTREAMS] [-n N] [-t TIME] "
                    "[-s SIZE] [-w WAIT] [-p PROF_MODE]\n"
                    "PROF_MODE = 0 : disable ABTX profiler\n"
                    "            1 : enable ABTX profiler (basic)\n"
                    "            2 : enable ABTX profiler (advanced)\n");
                return -1;
        }
    }
    if (num_xstreams <= 1) {
        printf("NUM_XSTERAMS (=`%d`) must be larger than 1.\n", num_xstreams);
        return -1;
    }
    if (n <= 0) {
        printf("N (=`%d`) must be larger than 0.\n", n);
        return -1;
    }

    /* Allocate memory */
    ABT_xstream primary_xstream;
    ABT_xstream *engine_xstreams =
        (ABT_xstream *)malloc(sizeof(ABT_xstream) * (num_xstreams - 1));
    ABT_sched *engine_scheds =
        (ABT_sched *)malloc(sizeof(ABT_sched) * (num_xstreams - 1));
    operation_arg_t *ops =
        (operation_arg_t *)malloc(THREAD_POOL_SIZE * sizeof(operation_arg_t));

    /* Initialize Argobots. */
    ABT_init(argc, argv);

    /* Initialize the profiler. */
    int prof_init = 0;
    ABTX_prof_context prof_context;
    prof_init = ABTX_prof_init(&prof_context);

    /* Set up pools */
    ABT_pool primary_pool, engine_pool;
    ABT_xstream_self(&primary_xstream);
    ABT_xstream_get_main_pools(primary_xstream, 1, &primary_pool);
    ABT_pool_create_basic(g_wait_mode == 0 ? ABT_POOL_FIFO : ABT_POOL_FIFO_WAIT,
                          ABT_POOL_ACCESS_MPMC, ABT_TRUE, &engine_pool);

    /* Create schedulers. */
    ABT_sched_def engine_sched_def = { .type = ABT_SCHED_TYPE_ULT,
                                       .init = sched_init,
                                       .run = sched_run,
                                       .free = sched_free,
                                       .get_migr_pool = NULL };
    for (i = 0; i < num_xstreams - 1; i++) {
        ABT_sched_create(&engine_sched_def, 1, &engine_pool,
                         ABT_SCHED_CONFIG_NULL, &engine_scheds[i]);
    }

    /* Create execution streams */
    for (i = 0; i < num_xstreams - 1; i++) {
        ABT_xstream_create(engine_scheds[i], &engine_xstreams[i]);
    }

    /* Main kernel */
    int step;
    for (step = 0; step < 2; step++) {
        /* The first step is for warm up. */

        /* Clean up arguments. */
        for (i = 0; i < THREAD_POOL_SIZE; i++) {
            ops[i].thread = ABT_THREAD_NULL;
        }
        /* Start a profiler. */
        if (prof_init == ABT_SUCCESS && prof_mode == 1) {
            ABTX_prof_start(prof_context, ABTX_PROF_MODE_BASIC);
        } else if (prof_init == ABT_SUCCESS && prof_mode == 2) {
            ABTX_prof_start(prof_context, ABTX_PROF_MODE_DETAILED);
        }

        /* Create total_ops threads in duration [s]. */
        int num_completed_ops = 0;
        int num_created_ops = 0;
        int num_total_ops = (step == 0) ? THREAD_POOL_SIZE : n;
        if (num_total_ops >= n)
            num_total_ops = n;
        double interval_per_ops =
            (step == 0) ? 0.0 : (duration / num_total_ops);

        double creation_end_time = 0.0;
        double start_time = ABT_get_wtime();
        while (num_completed_ops < num_total_ops) {
            int create_flag = 0;
            for (i = 0; i < THREAD_POOL_SIZE; i++) {
                /* Check if the primary thread should create a new operation */
                if (!create_flag && num_created_ops < num_total_ops) {
                    if (ABT_get_wtime() - start_time >=
                        interval_per_ops * (num_created_ops + 1)) {
                        create_flag = 1;
                    }
                }
                if (ops[i].thread != ABT_THREAD_NULL) {
                    /* This operation might be still running.  If it has been
                     * finished, let's free it. */
                    ABT_thread_state state;
                    ABT_thread_get_state(ops[i].thread, &state);
                    if (state == ABT_THREAD_STATE_TERMINATED) {
                        ABT_thread_free(&ops[i].thread);
                        /* ABT_THREAD_NULL is set to ops[i].thread in
                         * ABT_thread_free(). */
                        num_completed_ops++;
                    }
                }
                if (create_flag && ops[i].thread == ABT_THREAD_NULL) {
                    /* Create a new thread. */
                    ops[i].size = size;
                    ABT_thread_create(engine_pool, operation, &ops[i],
                                      ABT_THREAD_ATTR_NULL, &ops[i].thread);
                    create_flag = 0;
                    num_created_ops++;
                    if (num_created_ops == num_total_ops)
                        creation_end_time = ABT_get_wtime();
                }
            }
        }
        double end_time = ABT_get_wtime();

        if (prof_init == ABT_SUCCESS && (prof_mode == 1 || prof_mode == 2)) {
            ABTX_prof_stop(prof_context);
        }
        if (step != 0) {
            /* Print the result. */
            printf("##############################\n");
            printf("[%d] elapsed time = %f [s]\n", step, end_time - start_time);
            printf("[%d] creation throughput = %f [ops/s]\n", step,
                   num_total_ops / (creation_end_time - start_time));
            printf("[%d] engine throughput = %f [ops/s]\n", step,
                   num_total_ops / (end_time - start_time));
            /* Quickly measure the operation granularity. */
            int num_computes = 64 > n ? 64 : n;
            double compute_start_time = ABT_get_wtime();
            for (i = 0; i < num_computes; i++) {
                compute(size);
            }
            double compute_end_time = ABT_get_wtime();
            printf("[%d] approx. operation granularity = %f [us]\n", step,
                   (compute_end_time - compute_start_time) / num_computes *
                       1.0e6);

            if (prof_init == ABT_SUCCESS &&
                (prof_mode == 1 || prof_mode == 2)) {
                ABTX_prof_print(prof_context, stdout,
                                ABTX_PRINT_MODE_SUMMARY |
                                    ABTX_PRINT_MODE_FANCY);
            }
        }
    }

    /* Join secondary execution streams. */
    for (i = 0; i < num_xstreams - 1; i++) {
        ABT_xstream_join(engine_xstreams[i]);
        ABT_xstream_free(&engine_xstreams[i]);
    }

    /* Finalize the profiler. */
    if (prof_init == ABT_SUCCESS) {
        ABTX_prof_finalize(prof_context);
    }

    /* Free schedulers */
    /* Note that we do not need to free the scheduler for the primary ES,
     * because its scheduler will be automatically freed in ABT_finalize(). */
    for (i = 0; i < num_xstreams - 1; i++) {
        ABT_sched_free(&engine_scheds[i]);
    }

    /* Finalize Argobots. */
    ABT_finalize();

    free(engine_xstreams);
    free(engine_scheds);
    free(ops);

    return 0;
}
