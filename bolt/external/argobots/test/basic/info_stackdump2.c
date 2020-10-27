/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_NUM_THREADS 8

volatile int g_counter = 0;
volatile int g_stop = 0;

void callback_f(ABT_bool timeout, void *arg)
{
    assert(timeout == ABT_TRUE);
    assert((intptr_t)arg == (intptr_t)1);
    ATS_atomic_store(&g_stop, 0);
}

void signal_handler(int sig)
{
    /* In this test, some execution streams busy-wait in thread_func, so
     * timeout is set to a positive value so that this mechanism works even if
     * some execution streams hang.  */
    ABT_info_trigger_print_all_thread_stacks(stdout, 3.0, callback_f,
                                             (void *)((intptr_t)1));
    signal(SIGUSR1, signal_handler);
}

typedef struct thread_arg {
    int id;
    int issue_signal;
    int stop;
    int num_xstreams;
    int *dummy_ptr;
    void *stack;
} thread_arg_t;

void thread_func(void *arg)
{
    int i;
    int rank;
    ABT_unit_id id;
    ABT_thread self;
    thread_arg_t *t_arg = (thread_arg_t *)arg;

    ABT_xstream_self_rank(&rank);
    ABT_thread_self(&self);
    ABT_thread_get_id(self, &id);
    t_arg->id = (int)id;

    ATS_printf(1, "[U%d:E%d] START\n", t_arg->id, rank);

    /* Write data into a stack */
    int stack_data[128];
    t_arg->dummy_ptr = stack_data;
    for (i = 0; i < 128; i += 2) {
        t_arg->dummy_ptr[i + 0] = id;
        t_arg->dummy_ptr[i + 1] = rank;
    }

    ATS_printf(1, "[U%d:E%d] END\n", t_arg->id, rank);

    ABT_thread_yield();

    if (t_arg->issue_signal) {
        ATS_printf(1, "[U%d:E%d] Raise SIGUSR1\n", t_arg->id, rank);
        ATS_atomic_store(&g_stop, 1);
        while (ATS_atomic_load(&g_counter) < t_arg->num_xstreams - 2) {
            /* ensure that all the secondary execution streams are waiting in
             * the following busy loop. */
        }
        raise(SIGUSR1);
    }

    if (t_arg->stop) {
        ATS_atomic_fetch_add(&g_counter, 1);
        while (ATS_atomic_load(&g_stop) == 1)
            ;
    }
}

int main(int argc, char *argv[])
{
    int i, j;
    int ret;
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    int num_threads = DEFAULT_NUM_THREADS;

    /* Set a signal handler */
    signal(SIGUSR1, signal_handler);

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc >= 2) {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
        num_threads = ATS_get_arg_val(ATS_ARG_N_ULT);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(2, "# of ESs : %d\n", num_xstreams);
    ATS_printf(1, "# of ULTs: %d\n", num_threads);

    ABT_xstream *xstreams;
    ABT_thread **threads;
    thread_arg_t **args;
    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    assert(xstreams != NULL);
    threads = (ABT_thread **)malloc(sizeof(ABT_thread *) * num_xstreams);
    assert(threads != NULL);
    args = (thread_arg_t **)malloc(sizeof(thread_arg_t *) * num_xstreams);
    assert(args != NULL);
    for (i = 0; i < num_xstreams; i++) {
        threads[i] = (ABT_thread *)malloc(sizeof(ABT_thread) * num_threads);
        args[i] = (thread_arg_t *)malloc(sizeof(thread_arg_t) * num_threads);
    }

    /* Create execution streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the pools attached to an execution stream */
    ABT_pool *pools;
    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, pools + i);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create ULTs. */
    /* i starts from 1 because one execution stream must be busy-scheduling. */
    for (i = 1; i < num_xstreams; i++) {
        for (j = 0; j < num_threads; j++) {
            int tid = i * num_threads + j + 1;
            args[i][j].id = tid;
            args[i][j].issue_signal =
                (i == ((num_xstreams - 1) / 2) + 1) && (j == num_threads / 2);
            args[i][j].stop = (j == num_threads / 2);
            args[i][j].num_xstreams = num_xstreams;
            args[i][j].dummy_ptr = NULL;
            args[i][j].stack = NULL;
            if (tid % 3 == 0) {
                ret = ABT_thread_create(pools[i], thread_func,
                                        (void *)&args[i][j],
                                        ABT_THREAD_ATTR_NULL, &threads[i][j]);
                ATS_ERROR(ret, "ABT_thread_create");
            } else if (tid % 3 == 1) {
                ABT_thread_attr attr;
                ret = ABT_thread_attr_create(&attr);
                ATS_ERROR(ret, "ABT_thread_attr_create");
                ret = ABT_thread_attr_set_stacksize(attr, 32768);
                ATS_ERROR(ret, "ABT_thread_attr_set_stacksize");
                ret = ABT_thread_create(pools[i], thread_func,
                                        (void *)&args[i][j], attr,
                                        &threads[i][j]);
                ATS_ERROR(ret, "ABT_thread_create");
                ret = ABT_thread_attr_free(&attr);
                ATS_ERROR(ret, "ABT_thread_attr_free");
            } else {
                const size_t stacksize = 32768;
                args[i][j].stack = malloc(stacksize);
                ABT_thread_attr attr;
                ret = ABT_thread_attr_create(&attr);
                ATS_ERROR(ret, "ABT_thread_attr_create");
                ret = ABT_thread_attr_set_stack(attr, args[i][j].stack,
                                                stacksize);
                ATS_ERROR(ret, "ABT_thread_attr_set_stack");
                ret = ABT_thread_create(pools[i], thread_func,
                                        (void *)&args[i][j], attr,
                                        &threads[i][j]);
                ATS_ERROR(ret, "ABT_thread_create");
                ret = ABT_thread_attr_free(&attr);
                ATS_ERROR(ret, "ABT_thread_attr_free");
            }
        }
    }

    /* Join and free ULTs */
    for (i = 1; i < num_xstreams; i++) {
        for (j = 0; j < num_threads; j++) {
            ret = ABT_thread_free(&threads[i][j]);
            ATS_ERROR(ret, "ABT_thread_free");
        }
    }

    /* Join Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
    }

    /* Free Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Finalize */
    ret = ATS_finalize(0);

    for (i = 1; i < num_xstreams; i++) {
        for (j = 0; j < num_threads; j++) {
            if (args[i][j].stack)
                free(args[i][j].stack);
        }
    }
    for (i = 0; i < num_xstreams; i++) {
        free(threads[i]);
        free(args[i]);
    }
    free(threads);
    free(args);
    free(xstreams);
    free(pools);

    return ret;
}
