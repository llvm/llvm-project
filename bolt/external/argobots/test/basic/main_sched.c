/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "abt.h"
#include "abttest.h"

/* This test checks if main schedulers work as normal threads. */

#define DEFAULT_NUM_XSTREAMS 4
#define NUM_LOCKS 10

static int num_xstreams = DEFAULT_NUM_XSTREAMS;
ABT_barrier g_barrier;
ABT_mutex g_mutex;
static int g_val = 0;

static int sched_init(ABT_sched sched, ABT_sched_config config)
{
    return ABT_SUCCESS;
}

static void sched_run(ABT_sched sched)
{
    int ret, i, rank;
    ABT_pool pool;

    ret = ABT_sched_get_pools(sched, 1, 0, &pool);
    ATS_ERROR(ret, "ABT_sched_get_pools");
    ret = ABT_xstream_self_rank(&rank);
    ATS_ERROR(ret, "ABT_xstream_self_rank");

    /* The main scheduler can yield. */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_thread_yield();
        ATS_ERROR(ret, "ABT_thread_yield");
    }

    /* The main scheduler can barrier. */
    for (i = 0; i < num_xstreams; i++) {
        printf("END1! %d - %d\n", i, rank);
        ret = ABT_barrier_wait(g_barrier);
        ATS_ERROR(ret, "ABT_barrier_wait");
        if (i == rank)
            g_val++;
    }
    printf("END1! %d\n", rank);
    ret = ABT_barrier_wait(g_barrier);
    ATS_ERROR(ret, "ABT_barrier_wait");

    /* The main scheduler can lock a mutex. */
    for (i = 0; i < NUM_LOCKS; i++) {
        ret = ABT_mutex_lock(g_mutex);
        ATS_ERROR(ret, "ABT_mutex_lock");
        g_val++;
        ret = ABT_mutex_unlock(g_mutex);
        ATS_ERROR(ret, "ABT_mutex_unlock");
    }
    printf("END2! %d\n", rank);
    while (1) {
        /* Normal scheduling. */
        ABT_unit unit;
        ABT_bool stop;
        ABT_pool_pop(pool, &unit);
        if (unit != ABT_UNIT_NULL) {
            ABT_xstream_run_unit(unit, pool);
        }
        ret = ABT_sched_has_to_stop(sched, &stop);
        ATS_ERROR(ret, "ABT_sched_has_to_stop");
        if (stop == ABT_TRUE)
            break;
        ret = ABT_xstream_check_events(sched);
        ATS_ERROR(ret, "ABT_xstream_check_events");
    }
}

static int sched_free(ABT_sched sched)
{
    return ABT_SUCCESS;
}

int main(int argc, char *argv[])
{
    int i, ret;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc > 1) {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "num_xstreams=%d\n", num_xstreams);

    ABT_xstream *xstreams =
        (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);

    /* Create pools */
    ABT_pool *pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC,
                                    ABT_TRUE, &pools[i]);
        ATS_ERROR(ret, "ABT_pool_create_basic");
    }

    /* Create schedulers */
    ABT_sched_def sched_def = { .type = ABT_SCHED_TYPE_ULT,
                                .init = sched_init,
                                .run = sched_run,
                                .free = sched_free,
                                .get_migr_pool = NULL };

    ABT_sched *scheds = (ABT_sched *)malloc(sizeof(ABT_sched) * num_xstreams);
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_sched_create(&sched_def, 1, &pools[i], ABT_SCHED_CONFIG_NULL,
                               &scheds[i]);
        ATS_ERROR(ret, "ABT_sched_create");
    }

    /* Create a barrier. */
    ret = ABT_barrier_create(num_xstreams, &g_barrier);
    ATS_ERROR(ret, "ABT_barrier_create");

    /* Create a mutex */
    ret = ABT_mutex_create(&g_mutex);
    ATS_ERROR(ret, "ABT_mutex_create");

    /* Create execution streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(scheds[i], &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }
    /* Update the main scheduler of the primary execution stream after creating
     * the other execution streams. */
    ret = ABT_xstream_set_main_sched(xstreams[0], scheds[0]);
    ATS_ERROR(ret, "ABT_xstream_set_main_sched");

    /* Join and free the execution streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }
    /* Free the barrier */
    ret = ABT_barrier_free(&g_barrier);
    ATS_ERROR(ret, "ABT_barrier_free");

    /* Free the mutex */
    ret = ABT_mutex_free(&g_mutex);
    ATS_ERROR(ret, "ABT_mutex_free");

    /* Free the schedulers.  Note that we do not need to free the scheduler for
     * the primary ES, i.e., xstreams[0], because its scheduler will be freed
     * automatically in ABT_finalize(). */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_sched_free(&scheds[i]);
        ATS_ERROR(ret, "ABT_sched_free");
    }

    /* Finalize */
    ret = ATS_finalize(0);
    ATS_ERROR(ret, "ATS_finalize");

    free(scheds);
    free(xstreams);
    free(pools);

    /* Check g_val */
    assert(g_val == num_xstreams * (NUM_LOCKS + 1));
    return ABT_SUCCESS;
}
