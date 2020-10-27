/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

/** @defgroup SCHED_BASIC_WAIT Basic waiting scheduler
 * This group is for the basic waiting scheduler.
 */

static int sched_init(ABT_sched sched, ABT_sched_config config);
static void sched_run(ABT_sched sched);
static int sched_free(ABT_sched);
static void sched_sort_pools(int num_pools, ABT_pool *pools);

static ABT_sched_def sched_basic_wait_def = {
    .type = ABT_SCHED_TYPE_TASK,
    .init = sched_init,
    .run = sched_run,
    .free = sched_free,
    .get_migr_pool = NULL,
};

typedef struct {
    uint32_t event_freq;
    int num_pools;
    ABT_pool *pools;
} sched_data;

ABT_sched_config_var ABT_sched_basic_wait_freq = { .idx = 0,
                                                   .type =
                                                       ABT_SCHED_CONFIG_INT };

ABT_sched_def *ABTI_sched_get_basic_wait_def(void)
{
    return &sched_basic_wait_def;
}

static inline sched_data *sched_data_get_ptr(void *data)
{
    return (sched_data *)data;
}

static int sched_init(ABT_sched sched, ABT_sched_config config)
{
    int abt_errno;
    int num_pools;

    ABTI_sched *p_sched = ABTI_sched_get_ptr(sched);
    ABTI_CHECK_NULL_SCHED_PTR(p_sched);

    /* Default settings */
    sched_data *p_data;
    abt_errno = ABTU_malloc(sizeof(sched_data), (void **)&p_data);
    ABTI_CHECK_ERROR(abt_errno);

    p_data->event_freq = gp_ABTI_global->sched_event_freq;

    /* Set the variables from the config */
    void *p_event_freq = &p_data->event_freq;
    abt_errno = ABTI_sched_config_read(config, 1, 1, &p_event_freq);
    if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
        ABTU_free(p_data);
        ABTI_CHECK_ERROR(abt_errno);
    }

    /* Save the list of pools */
    num_pools = p_sched->num_pools;
    p_data->num_pools = num_pools;
    abt_errno =
        ABTU_malloc(num_pools * sizeof(ABT_pool), (void **)&p_data->pools);
    if (ABTI_IS_ERROR_CHECK_ENABLED && abt_errno != ABT_SUCCESS) {
        ABTU_free(p_data);
        ABTI_CHECK_ERROR(abt_errno);
    }
    memcpy(p_data->pools, p_sched->pools, sizeof(ABT_pool) * num_pools);

    /* Sort pools according to their access mode so the scheduler can execute
       work units from the private pools. */
    if (num_pools > 1) {
        sched_sort_pools(num_pools, p_data->pools);
    }

    p_sched->data = p_data;
    return ABT_SUCCESS;
}

static void sched_run(ABT_sched sched)
{
    ABTI_xstream *p_local_xstream =
        ABTI_local_get_xstream(ABTI_local_get_local());
    uint32_t work_count = 0;
    sched_data *p_data;
    uint32_t event_freq;
    int num_pools;
    ABT_pool *pools;
    int i;
    int run_cnt_nowait;

    ABTI_sched *p_sched = ABTI_sched_get_ptr(sched);
    ABTI_ASSERT(p_sched);

    p_data = sched_data_get_ptr(p_sched->data);
    event_freq = p_data->event_freq;
    num_pools = p_data->num_pools;
    pools = p_data->pools;

    while (1) {
        run_cnt_nowait = 0;

        /* Execute one work unit from the scheduler's pool */
        for (i = 0; i < num_pools; i++) {
            ABT_pool pool = pools[i];
            ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
            /* Pop one work unit */
            ABT_unit unit = ABTI_pool_pop(p_pool);
            if (unit != ABT_UNIT_NULL) {
                ABTI_xstream_run_unit(&p_local_xstream, unit, p_pool);
                run_cnt_nowait++;
                break;
            }
        }

        /* Block briefly on pop_wait() if we didn't find work to do in main loop
         * above. */
        if (!run_cnt_nowait) {
            ABT_unit unit =
                ABTI_pool_pop_wait(ABTI_pool_get_ptr(pools[0]), 0.1);
            if (unit != ABT_UNIT_NULL) {
                ABTI_xstream_run_unit(&p_local_xstream, unit,
                                      ABTI_pool_get_ptr(pools[0]));
                break;
            }
        }

        /* If run_cnt_nowait is zero, that means that no units were found in
         * first pass through pools and we must have called pop_wait above.  We
         * should check events regardless of work_count in that case for them to
         * be processed in a timely manner. */
        if (!run_cnt_nowait || (++work_count >= event_freq)) {
            ABTI_xstream_check_events(p_local_xstream, p_sched);
            ABTI_local *p_local = ABTI_xstream_get_local(p_local_xstream);
            if (ABTI_sched_has_to_stop(&p_local, p_sched) == ABT_TRUE)
                break;
            p_local_xstream = ABTI_local_get_xstream(p_local);
            work_count = 0;
        }
    }
}

static int sched_free(ABT_sched sched)
{
    ABTI_sched *p_sched = ABTI_sched_get_ptr(sched);
    ABTI_ASSERT(p_sched);

    sched_data *p_data = sched_data_get_ptr(p_sched->data);
    ABTU_free(p_data->pools);
    ABTU_free(p_data);
    return ABT_SUCCESS;
}

static int pool_get_access_num(ABT_pool *p_pool)
{
    ABT_pool_access access;
    int num = 0;

    access = ABTI_pool_get_ptr(*p_pool)->access;
    switch (access) {
        case ABT_POOL_ACCESS_PRIV:
            num = 0;
            break;
        case ABT_POOL_ACCESS_SPSC:
        case ABT_POOL_ACCESS_MPSC:
            num = 1;
            break;
        case ABT_POOL_ACCESS_SPMC:
        case ABT_POOL_ACCESS_MPMC:
            num = 2;
            break;
        default:
            ABTI_ASSERT(0);
            ABTU_unreachable();
    }

    return num;
}

static int sched_cmp_pools(const void *p1, const void *p2)
{
    int p1_access, p2_access;

    p1_access = pool_get_access_num((ABT_pool *)p1);
    p2_access = pool_get_access_num((ABT_pool *)p2);

    if (p1_access > p2_access) {
        return 1;
    } else if (p1_access < p2_access) {
        return -1;
    } else {
        return 0;
    }
}

static void sched_sort_pools(int num_pools, ABT_pool *pools)
{
    qsort(pools, num_pools, sizeof(ABT_pool), sched_cmp_pools);
}
