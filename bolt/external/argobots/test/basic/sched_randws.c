/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_NUM_THREADS 4

static int num_threads = DEFAULT_NUM_THREADS;

static int g_counter = 0;
static ABT_mutex g_mutex = ABT_MUTEX_NULL;

static void thread_func(void *arg)
{
    ATS_UNUSED(arg);
    int old_rank, cur_rank;
    ABT_thread self;
    ABT_unit_id id;
    char *msg;

    ABT_xstream_self_rank(&cur_rank);
    ABT_thread_self(&self);
    ABT_thread_get_id(self, &id);

    ATS_printf(1, "[U%lu:E%d] Hello, world!\n", id, cur_rank);

    ABT_thread_yield();

    old_rank = cur_rank;
    ABT_xstream_self_rank(&cur_rank);
    msg = (cur_rank == old_rank) ? "" : " (stolen)";
    ATS_printf(1, "[U%lu:E%d] Hello again #1.%s\n", id, cur_rank, msg);

    ABT_thread_yield();

    old_rank = cur_rank;
    ABT_xstream_self_rank(&cur_rank);
    msg = (cur_rank == old_rank) ? "" : " (stolen)";
    ATS_printf(1, "[U%lu:E%d] Hello again #2.%s\n", id, cur_rank, msg);

    ABT_thread_yield();

    old_rank = cur_rank;
    ABT_xstream_self_rank(&cur_rank);
    msg = (cur_rank == old_rank) ? "" : " (stolen)";
    ATS_printf(1, "[U%lu:E%d] Goodbye, world!%s\n", id, cur_rank, msg);

    ABT_mutex_lock(g_mutex);
    g_counter++;
    ABT_mutex_unlock(g_mutex);
}

static void create_threads(void *arg)
{
    ATS_UNUSED(arg);
    int i, rank;
    int ret;
    ABT_xstream xstream;
    ABT_pool pool;
    ABT_thread self;
    ABT_thread *threads;
    ABT_unit_id id;

    ret = ABT_xstream_self(&xstream);
    ATS_ERROR(ret, "ABT_xstream_self");
    ret = ABT_xstream_get_main_pools(xstream, 1, &pool);
    ATS_ERROR(ret, "ABT_xstream_get_main_pools");

    ABT_xstream_get_rank(xstream, &rank);
    ABT_thread_self(&self);
    ABT_thread_get_id(self, &id);

    ATS_printf(1, "[U%lu:E%d] creating ULTs\n", id, rank);
    threads = (ABT_thread *)malloc(sizeof(ABT_thread) * num_threads);
    for (i = 0; i < num_threads; i++) {
        ret = ABT_thread_create(pool, thread_func, NULL, ABT_THREAD_ATTR_NULL,
                                &threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    ABT_xstream_get_rank(xstream, &rank);
    ATS_printf(1, "[U%lu:E%d] freeing ULTs\n", id, rank);
    for (i = 0; i < num_threads; i++) {
        ret = ABT_thread_free(&threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
    }
    free(threads);
}

int main(int argc, char *argv[])
{
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    ABT_xstream *xstreams;
    ABT_sched *scheds;
    ABT_pool *pools, *my_pools;
    ABT_thread *main_threads;
    int i, k, ret;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc > 1) {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
        num_threads = ATS_get_arg_val(ATS_ARG_N_ULT);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1,
               "# of ESs    : %d\n"
               "# of ULTs/ES: %d\n",
               num_xstreams, num_threads);

    xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    scheds = (ABT_sched *)malloc(num_xstreams * sizeof(ABT_sched));
    pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    main_threads = (ABT_thread *)malloc(num_xstreams * sizeof(ABT_thread));

    /* Create a mutex */
    ret = ABT_mutex_create(&g_mutex);
    ATS_ERROR(ret, "ABT_mutex_create");

    /* Create pools */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC,
                                    ABT_TRUE, &pools[i]);
        ATS_ERROR(ret, "ABT_pool_create_basic");
    }

    /* Create schedulers */
    my_pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    for (i = 0; i < num_xstreams; i++) {
        for (k = 0; k < num_xstreams; k++) {
            my_pools[k] = pools[(i + k) % num_xstreams];
        }

        ret = ABT_sched_create_basic(ABT_SCHED_RANDWS, num_xstreams, my_pools,
                                     ABT_SCHED_CONFIG_NULL, &scheds[i]);
        ATS_ERROR(ret, "ABT_sched_create_basic");
    }
    free(my_pools);

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    ret = ABT_xstream_set_main_sched(xstreams[0], scheds[0]);
    ATS_ERROR(ret, "ABT_xstream_set_main_sched");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(scheds[i], &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Create main ULTs */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_thread_create(pools[i], create_threads, NULL,
                                ABT_THREAD_ATTR_NULL, &main_threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    /* Join and free main ULTs */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_thread_free(&main_threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
    }

    /* Join and free Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Free the mutex */
    ret = ABT_mutex_free(&g_mutex);
    ATS_ERROR(ret, "ABT_mutex_free");

    /* Validation */
    int expected = num_xstreams * num_threads;
    if (g_counter != expected) {
        fprintf(stderr, "expected=%d vs. g_counter=%d\n", expected, g_counter);
    }
    assert(g_counter == expected);

    /* Finalize */
    ret = ATS_finalize(0);

    free(xstreams);
    free(scheds);
    free(pools);
    free(main_threads);

    return ret;
}
