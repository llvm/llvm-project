/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "abt.h"
#include "abttest.h"

/* This test creates a user-defined scheduler, which is a simple work-stealing
 * scheduler, and uses it for ESs. */

#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_NUM_THREADS 4

static int num_xstreams = DEFAULT_NUM_XSTREAMS;
static int num_threads = DEFAULT_NUM_THREADS;

/* scheduler data structure and functions */
typedef struct {
    uint32_t event_freq;
} sched_data_t;

static inline sched_data_t *sched_data_get_ptr(void *data)
{
    return (sched_data_t *)data;
}

static int sched_init(ABT_sched sched, ABT_sched_config config)
{
    int ret = ABT_SUCCESS;

    sched_data_t *p_data = (sched_data_t *)calloc(1, sizeof(sched_data_t));

    /* Set the variables from the config */
    ret = ABT_sched_config_read(config, 1, &p_data->event_freq);
    ATS_ERROR(ret, "ABT_sched_config_read");

    ret = ABT_sched_set_data(sched, (void *)p_data);

    return ret;
}

static void sched_run(ABT_sched sched)
{
    int ret = ABT_SUCCESS;
    uint32_t work_count = 0;
    void *data;
    sched_data_t *p_data;
    ABT_pool my_pool;
    size_t size;
    ABT_unit unit;
    uint32_t event_freq;
    int num_pools;
    ABT_pool *pools;
    int target;
    unsigned seed = time(NULL);

    ABT_sched_get_data(sched, &data);
    p_data = sched_data_get_ptr(data);
    event_freq = p_data->event_freq;

    ret = ABT_sched_get_num_pools(sched, &num_pools);
    ATS_ERROR(ret, "ABT_sched_get_num_pools");

    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_pools);
    ret = ABT_sched_get_pools(sched, num_pools, 0, pools);
    ATS_ERROR(ret, "ABT_sched_get_pools");
    my_pool = pools[0];

    while (1) {
        /* Execute one work unit from the scheduler's pool */
        ABT_pool_pop(my_pool, &unit);
        if (unit != ABT_UNIT_NULL) {
            ABT_xstream_run_unit(unit, my_pool);
        } else if (num_pools > 1) {
            /* Steal a work unit from other pools */
            target =
                (num_pools == 2) ? 1 : (rand_r(&seed) % (num_pools - 1) + 1);
            ABT_pool tar_pool = pools[target];
            ABT_pool_get_size(tar_pool, &size);
            if (size > 0) {
                /* Pop one work unit */
                ABT_pool_pop(tar_pool, &unit);
                if (unit != ABT_UNIT_NULL) {
                    ABT_xstream_run_unit(unit, tar_pool);
                }
            }
        }

        if (++work_count >= event_freq) {
            ABT_bool stop;
            ret = ABT_sched_has_to_stop(sched, &stop);
            ATS_ERROR(ret, "ABT_sched_has_to_stop");
            if (stop == ABT_TRUE)
                break;
            work_count = 0;
            ABT_xstream_check_events(sched);
        }
    }

    free(pools);
}

static int sched_free(ABT_sched sched)
{
    int ret = ABT_SUCCESS;
    void *data;

    ABT_sched_get_data(sched, &data);
    sched_data_t *p_data = sched_data_get_ptr(data);
    free(p_data);

    return ret;
}

static ABT_pool *create_pools(int num)
{
    ABT_pool *pools;
    int i, ret;

    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num);
    for (i = 0; i < num; i++) {
        ret = ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC,
                                    ABT_TRUE, &pools[i]);
        ATS_ERROR(ret, "ABT_pool_create_basic");
    }

    return pools;
}

static ABT_sched *create_scheds(int num, ABT_pool *pools)
{
    int i, k, ret;
    ABT_pool *my_pools;
    ABT_sched *scheds;
    ABT_sched_config config;

    ABT_sched_config_var cv_event_freq = { .idx = 0,
                                           .type = ABT_SCHED_CONFIG_INT };

    ABT_sched_def sched_def = { .type = ABT_SCHED_TYPE_ULT,
                                .init = sched_init,
                                .run = sched_run,
                                .free = sched_free,
                                .get_migr_pool = NULL };

    /* Create a scheduler config */
    /* NOTE: The same scheduler config can be used for all schedulers. */
    ret = ABT_sched_config_create(&config, cv_event_freq, 10,
                                  ABT_sched_config_var_end);
    ATS_ERROR(ret, "ABT_sched_config_create");

    my_pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num);
    scheds = (ABT_sched *)malloc(sizeof(ABT_sched) * num);
    for (i = 0; i < num; i++) {
        for (k = 0; k < num; k++) {
            my_pools[k] = pools[(i + k) % num];
        }

        ret = ABT_sched_create(&sched_def, num, my_pools, config, &scheds[i]);
        ATS_ERROR(ret, "ABT_sched_create");
    }
    free(my_pools);

    ret = ABT_sched_config_free(&config);
    ATS_ERROR(ret, "ABT_sched_config_free");

    return scheds;
}

static void free_scheds(int num, ABT_sched *scheds)
{
    int i, ret;

    /* Free schedulers */
    /* Note that we do not need to free the scheduler for the primary ES,
     * i.e., xstreams[0], because its scheduler will be automatically freed in
     * ABT_finalize(). */
    for (i = 1; i < num; i++) {
        ret = ABT_sched_free(&scheds[i]);
        ATS_ERROR(ret, "ABT_sched_free");
    }

    free(scheds);
}

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
    int i, ret;
    ABT_xstream *xstreams;
    ABT_sched *scheds;
    ABT_pool *pools;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc > 1) {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
        num_threads = ATS_get_arg_val(ATS_ARG_N_ULT);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "num_xstreams=%d num_threads=%d\n", num_xstreams,
               num_threads);

    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);

    /* Create pools */
    pools = create_pools(num_xstreams);

    /* Create schedulers */
    scheds = create_scheds(num_xstreams, pools);

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    ret = ABT_xstream_set_main_sched(xstreams[0], scheds[0]);
    ATS_ERROR(ret, "ABT_xstream_set_main_sched");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(scheds[i], &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Create ULTs */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_thread_create(pools[i], create_threads, NULL,
                                ABT_THREAD_ATTR_NULL, NULL);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    /* Join and free Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    free_scheds(num_xstreams, scheds);

    /* Finalize */
    ret = ATS_finalize(0);

    free(xstreams);
    free(pools);

    return ret;
}
