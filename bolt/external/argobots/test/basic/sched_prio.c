/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/* This code tests the predefined priority schedulers. */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_UNITS 6
int num_units = DEFAULT_NUM_UNITS;

ABT_pool_access accesses[] = { ABT_POOL_ACCESS_PRIV, ABT_POOL_ACCESS_SPSC,
                               ABT_POOL_ACCESS_MPSC, ABT_POOL_ACCESS_SPMC,
                               ABT_POOL_ACCESS_MPMC };

typedef struct {
    int *num_pools;
    ABT_pool **pools;
    ABT_sched *scheds;
    ABT_xstream *xstreams;
} g_data_t;

const int num_scheds = sizeof(accesses) / sizeof(accesses[0]) + 1;

static g_data_t g_data;

static void init_global_data(void);
static void fini_global_data(void);
static void create_scheds_and_xstreams(void);
static void free_scheds_and_xstreams(void);
static void create_work_units(void);

int main(int argc, char *argv[])
{
    int i, ret;

    if (argc > 1)
        num_units = atoi(argv[1]);
    assert(num_units >= 0);

    /* Initialize */
    ATS_read_args(argc, argv);
    ATS_init(argc, argv, num_scheds);

    /* Create schedulers and ESs */
    init_global_data();
    create_scheds_and_xstreams();

    /* Create work units */
    create_work_units();

    /* Join ESs */
    for (i = 1; i < num_scheds; i++) {
        ret = ABT_xstream_join(g_data.xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
    }

    free_scheds_and_xstreams();

    /* Finalize */
    ret = ATS_finalize(0);
    fini_global_data();

    return ret;
}

static void init_global_data(void)
{
    g_data.num_pools = (int *)calloc(num_scheds, sizeof(int));
    g_data.pools = (ABT_pool **)calloc(num_scheds, sizeof(ABT_pool *));
    g_data.scheds = (ABT_sched *)calloc(num_scheds, sizeof(ABT_sched));
    g_data.xstreams = (ABT_xstream *)calloc(num_scheds, sizeof(ABT_xstream));
}

static void fini_global_data(void)
{
    int i;

    for (i = 0; i < num_scheds; i++) {
        if (g_data.pools[i])
            free(g_data.pools[i]);
    }

    free(g_data.num_pools);
    free(g_data.pools);
    free(g_data.scheds);
    free(g_data.xstreams);
}

static void create_scheds_and_xstreams(void)
{
    int i, k, ret;
    ABT_sched *scheds = g_data.scheds;
    int *num_pools = g_data.num_pools;
    ABT_pool **pools = g_data.pools;
    ABT_xstream *xstreams = g_data.xstreams;

    for (i = 0; i < num_scheds; i++) {
        if (i == num_scheds - 1) {
            /* Create pools and then create a scheduler */
            num_pools[i] = 2;
            pools[i] = (ABT_pool *)malloc(num_pools[i] * sizeof(ABT_pool));
            pools[i][0] = ABT_POOL_NULL;
            for (k = 1; k < num_pools[i]; k++) {
                ret = ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPSC,
                                            ABT_TRUE, &pools[i][k]);
                ATS_ERROR(ret, "ABT_pool_create_basic");
            }

            ret = ABT_sched_create_basic(ABT_SCHED_PRIO, num_pools[i], pools[i],
                                         ABT_SCHED_CONFIG_NULL, &scheds[i]);
            ATS_ERROR(ret, "ABT_sched_create_basic");
        } else {
            /* Create a scheduler and then get the list of pools */
            ABT_sched_config config;
            ret =
                ABT_sched_config_create(&config, ABT_sched_config_access,
                                        accesses[i], ABT_sched_config_var_end);
            ATS_ERROR(ret, "ABT_sched_config_create");
            ret = ABT_sched_create_basic(ABT_SCHED_PRIO, 0, NULL, config,
                                         &scheds[i]);
            ATS_ERROR(ret, "ABT_sched_create_basic");
            ret = ABT_sched_config_free(&config);
            ATS_ERROR(ret, "ABT_sched_config_free");

            ret = ABT_sched_get_num_pools(scheds[i], &num_pools[i]);
            ATS_ERROR(ret, "ABT_sched_get_num_pools");
            pools[i] = (ABT_pool *)malloc(num_pools[i] * sizeof(ABT_pool));
        }
        ret = ABT_sched_get_pools(scheds[i], num_pools[i], 0, pools[i]);
        ATS_ERROR(ret, "ABT_sched_get_pools");

        /* Create ES */
        if (i == 0) {
            ret = ABT_xstream_self(&xstreams[i]);
            ATS_ERROR(ret, "ABT_xstream_self");
            ret = ABT_xstream_set_main_sched(xstreams[i], scheds[i]);
            ATS_ERROR(ret, "ABT_xstream_set_main_sched");
        } else {
            /* If the predefined scheduler is associated with PW pools,
               we will stack it so that the primary ULT can add the initial
               work unit. */
            if (i == num_scheds - 1 || accesses[i] == ABT_POOL_ACCESS_PRIV ||
                accesses[i] == ABT_POOL_ACCESS_SPSC ||
                accesses[i] == ABT_POOL_ACCESS_SPMC) {
                ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
                ATS_ERROR(ret, "ABT_xstream_create");
            } else {
                ret = ABT_xstream_create(scheds[i], &xstreams[i]);
                ATS_ERROR(ret, "ABT_xstream_create");
            }
        }
    }
}

static void free_scheds_and_xstreams(void)
{
    int i, ret;
    ABT_xstream *xstreams = g_data.xstreams;

    /* Free all ESs except the primary ES */
    for (i = 1; i < num_scheds; i++) {
        /* Free the ES */
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }
}

typedef struct {
    int es_id;
    int my_id;
    int prio;
} unit_arg_t;

static ABT_bool verify_exec_order(int es_id, int my_prio)
{
    if (my_prio == 0)
        return ABT_TRUE;

    ABT_pool *my_pools = g_data.pools[es_id];
    size_t pool_size;
    int i, ret;

    for (i = 0; i < my_prio; i++) {
        ret = ABT_pool_get_size(my_pools[i], &pool_size);
        ATS_ERROR(ret, "ABT_pool_get_size");
        if (pool_size > 0)
            return ABT_FALSE;
    }
    return ABT_TRUE;
}

static void thread_func(void *arg)
{
    unit_arg_t *my_arg = (unit_arg_t *)arg;
    ABT_bool valid;

    valid = verify_exec_order(my_arg->es_id, my_arg->prio);
    assert(valid == ABT_TRUE);
    ATS_printf(1, "[E%d:U%d:P%d] before yield\n", my_arg->es_id, my_arg->my_id,
               my_arg->prio);

    ABT_thread_yield();

    valid = verify_exec_order(my_arg->es_id, my_arg->prio);
    assert(valid == ABT_TRUE);
    ATS_printf(1, "[E%d:U%d:P%d] after yield\n", my_arg->es_id, my_arg->my_id,
               my_arg->prio);

    free(my_arg);
}

static void task_func(void *arg)
{
    unit_arg_t *my_arg = (unit_arg_t *)arg;
    ABT_bool valid;

    valid = verify_exec_order(my_arg->es_id, my_arg->prio);
    assert(valid == ABT_TRUE);
    ATS_printf(1, "[E%d:T%d:P%d] running\n", my_arg->es_id, my_arg->my_id,
               my_arg->prio);

    free(my_arg);
}

static void gen_work(void *arg)
{
    int idx = (size_t)arg;
    int num_pools = g_data.num_pools[idx];
    ABT_pool *my_pools = g_data.pools[idx];
    ABT_bool flag;
    int i, ret;
    unsigned seed = time(NULL);

    ATS_printf(1, "[E%d] creating work units\n", idx);

    /* Create work units on the current ES */
    /* TODO: add work units to pools of other ESs */
    for (i = 0; i < num_units; i++) {
        unit_arg_t *my_arg = (unit_arg_t *)malloc(sizeof(unit_arg_t));
        my_arg->es_id = idx;
        my_arg->my_id = i;
        my_arg->prio = rand_r(&seed) % num_pools;

        if (i & 1) {
            ret = ABT_thread_create(my_pools[my_arg->prio], thread_func,
                                    (void *)my_arg, ABT_THREAD_ATTR_NULL, NULL);
            ATS_ERROR(ret, "ABT_thread_create");
        } else {
            ret = ABT_task_create(my_pools[my_arg->prio], task_func,
                                  (void *)my_arg, NULL);
            ATS_ERROR(ret, "ABT_task_create");
        }
    }

    /* Stack the priority scheduler if it is associated with PW pools */
    ret = ABT_self_on_primary_xstream(&flag);
    ATS_ERROR(ret, "ABT_self_on_primary_stream");
    if (flag == ABT_FALSE) {
        if (idx == num_scheds - 1 || accesses[idx] == ABT_POOL_ACCESS_PRIV ||
            accesses[idx] == ABT_POOL_ACCESS_SPSC ||
            accesses[idx] == ABT_POOL_ACCESS_SPMC) {
            ABT_pool main_pool;
            ret =
                ABT_xstream_get_main_pools(g_data.xstreams[idx], 1, &main_pool);
            ATS_ERROR(ret, "ABT_xstream_get_main_pools");
            ret = ABT_pool_add_sched(main_pool, g_data.scheds[idx]);
            ATS_ERROR(ret, "ABT_pool_add_sched");
        }
    }
}

static void create_work_units(void)
{
    int i;
    int ret;

    for (i = 0; i < num_scheds; i++) {
        /* Create a ULT in the first main pool */
        ABT_pool main_pool;
        ret = ABT_xstream_get_main_pools(g_data.xstreams[i], 1, &main_pool);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");

        ret = ABT_thread_create(main_pool, gen_work, (void *)(size_t)i,
                                ABT_THREAD_ATTR_NULL, NULL);
        ATS_ERROR(ret, "ABT_thread_create");
    }
}
