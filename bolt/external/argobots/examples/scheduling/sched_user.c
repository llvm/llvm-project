/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "abt.h"

#define NUM_XSTREAMS 4
#define NUM_THREADS 4

static void create_scheds(int num, ABT_pool *pools, ABT_sched *scheds);
static void create_threads(void *arg);
static void thread_hello(void *arg);

int main(int argc, char *argv[])
{
    ABT_xstream xstreams[NUM_XSTREAMS];
    ABT_sched scheds[NUM_XSTREAMS];
    ABT_pool pools[NUM_XSTREAMS];
    ABT_thread threads[NUM_XSTREAMS];
    int i;

    ABT_init(argc, argv);

    /* Create pools */
    for (i = 0; i < NUM_XSTREAMS; i++) {
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE,
                              &pools[i]);
    }

    /* Create schedulers */
    create_scheds(NUM_XSTREAMS, pools, scheds);

    /* Create ESs */
    ABT_xstream_self(&xstreams[0]);
    ABT_xstream_set_main_sched(xstreams[0], scheds[0]);
    for (i = 1; i < NUM_XSTREAMS; i++) {
        ABT_xstream_create(scheds[i], &xstreams[i]);
    }

    /* Create ULTs */
    for (i = 0; i < NUM_XSTREAMS; i++) {
        size_t tid = (size_t)i;
        ABT_thread_create(pools[i], create_threads, (void *)tid,
                          ABT_THREAD_ATTR_NULL, &threads[i]);
    }

    /* Join & Free */
    for (i = 0; i < NUM_XSTREAMS; i++) {
        ABT_thread_join(threads[i]);
        ABT_thread_free(&threads[i]);
    }
    for (i = 1; i < NUM_XSTREAMS; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }

    /* Free schedulers */
    /* Note that we do not need to free the scheduler for the primary ES,
     * i.e., xstreams[0], because its scheduler will be automatically freed in
     * ABT_finalize(). */
    for (i = 1; i < NUM_XSTREAMS; i++) {
        ABT_sched_free(&scheds[i]);
    }

    /* Finalize */
    ABT_finalize();

    return 0;
}

/******************************************************************************/
/* Scheduler data structure and functions                                     */
/******************************************************************************/
typedef struct {
    uint32_t event_freq;
} sched_data_t;

static int sched_init(ABT_sched sched, ABT_sched_config config)
{
    sched_data_t *p_data = (sched_data_t *)calloc(1, sizeof(sched_data_t));

    ABT_sched_config_read(config, 1, &p_data->event_freq);
    ABT_sched_set_data(sched, (void *)p_data);

    return ABT_SUCCESS;
}

static void sched_run(ABT_sched sched)
{
    uint32_t work_count = 0;
    sched_data_t *p_data;
    int num_pools;
    ABT_pool *pools;
    ABT_unit unit;
    int target;
    ABT_bool stop;
    unsigned seed = time(NULL);

    ABT_sched_get_data(sched, (void **)&p_data);
    ABT_sched_get_num_pools(sched, &num_pools);
    pools = (ABT_pool *)malloc(num_pools * sizeof(ABT_pool));
    ABT_sched_get_pools(sched, num_pools, 0, pools);

    while (1) {
        /* Execute one work unit from the scheduler's pool */
        ABT_pool_pop(pools[0], &unit);
        if (unit != ABT_UNIT_NULL) {
            ABT_xstream_run_unit(unit, pools[0]);
        } else if (num_pools > 1) {
            /* Steal a work unit from other pools */
            target =
                (num_pools == 2) ? 1 : (rand_r(&seed) % (num_pools - 1) + 1);
            ABT_pool_pop(pools[target], &unit);
            if (unit != ABT_UNIT_NULL) {
                ABT_xstream_run_unit(unit, pools[target]);
            }
        }

        if (++work_count >= p_data->event_freq) {
            work_count = 0;
            ABT_sched_has_to_stop(sched, &stop);
            if (stop == ABT_TRUE)
                break;
            ABT_xstream_check_events(sched);
        }
    }

    free(pools);
}

static int sched_free(ABT_sched sched)
{
    sched_data_t *p_data;

    ABT_sched_get_data(sched, (void **)&p_data);
    free(p_data);

    return ABT_SUCCESS;
}

static void create_scheds(int num, ABT_pool *pools, ABT_sched *scheds)
{
    ABT_sched_config config;
    ABT_pool *my_pools;
    int i, k;

    ABT_sched_config_var cv_event_freq = { .idx = 0,
                                           .type = ABT_SCHED_CONFIG_INT };

    ABT_sched_def sched_def = { .type = ABT_SCHED_TYPE_ULT,
                                .init = sched_init,
                                .run = sched_run,
                                .free = sched_free,
                                .get_migr_pool = NULL };

    /* Create a scheduler config */
    ABT_sched_config_create(&config, cv_event_freq, 10,
                            ABT_sched_config_var_end);

    my_pools = (ABT_pool *)malloc(num * sizeof(ABT_pool));
    for (i = 0; i < num; i++) {
        for (k = 0; k < num; k++) {
            my_pools[k] = pools[(i + k) % num];
        }

        ABT_sched_create(&sched_def, num, my_pools, config, &scheds[i]);
    }
    free(my_pools);

    ABT_sched_config_free(&config);
}

static void create_threads(void *arg)
{
    int i, rank, tid = (int)(size_t)arg;
    ABT_xstream xstream;
    ABT_pool pool;
    ABT_thread *threads;

    ABT_xstream_self(&xstream);
    ABT_xstream_get_main_pools(xstream, 1, &pool);

    ABT_xstream_get_rank(xstream, &rank);
    printf("[U%d:E%d] creating ULTs\n", tid, rank);

    threads = (ABT_thread *)malloc(sizeof(ABT_thread) * NUM_THREADS);
    for (i = 0; i < NUM_THREADS; i++) {
        size_t id = (rank + 1) * 10 + i;
        ABT_thread_create(pool, thread_hello, (void *)id, ABT_THREAD_ATTR_NULL,
                          &threads[i]);
    }

    ABT_xstream_get_rank(xstream, &rank);
    printf("[U%d:E%d] freeing ULTs\n", tid, rank);
    for (i = 0; i < NUM_THREADS; i++) {
        ABT_thread_free(&threads[i]);
    }
    free(threads);
}

static void thread_hello(void *arg)
{
    int tid = (int)(size_t)arg;
    int old_rank, cur_rank;
    char *msg;

    ABT_xstream_self_rank(&cur_rank);

    printf("  [U%d:E%d] Hello, world!\n", tid, cur_rank);

    ABT_thread_yield();

    old_rank = cur_rank;
    ABT_xstream_self_rank(&cur_rank);
    msg = (cur_rank == old_rank) ? "" : " (stolen)";
    printf("  [U%d:E%d] Hello again.%s\n", tid, cur_rank, msg);

    ABT_thread_yield();

    old_rank = cur_rank;
    ABT_xstream_self_rank(&cur_rank);
    msg = (cur_rank == old_rank) ? "" : " (stolen)";
    printf("  [U%d:E%d] Goodbye, world!%s\n", tid, cur_rank, msg);
}
