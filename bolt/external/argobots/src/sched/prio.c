/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

/* Priority Scheduler Implementation */

static int sched_init(ABT_sched sched, ABT_sched_config config);
static void sched_run(ABT_sched sched);
static int sched_free(ABT_sched);

static ABT_sched_def sched_prio_def = { .type = ABT_SCHED_TYPE_TASK,
                                        .init = sched_init,
                                        .run = sched_run,
                                        .free = sched_free,
                                        .get_migr_pool = NULL };

typedef struct {
    uint32_t event_freq;
    int num_pools;
    ABT_pool *pools;
#ifdef ABT_CONFIG_USE_SCHED_SLEEP
    struct timespec sleep_time;
#endif
} sched_data;

ABT_sched_def *ABTI_sched_get_prio_def(void)
{
    return &sched_prio_def;
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
#ifdef ABT_CONFIG_USE_SCHED_SLEEP
    p_data->sleep_time.tv_sec = 0;
    p_data->sleep_time.tv_nsec = gp_ABTI_global->sched_sleep_nsec;
#endif

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
    CNT_DECL(run_cnt);

    ABTI_sched *p_sched = ABTI_sched_get_ptr(sched);
    ABTI_ASSERT(p_sched);

    p_data = sched_data_get_ptr(p_sched->data);
    event_freq = p_data->event_freq;
    num_pools = p_sched->num_pools;
    pools = p_data->pools;

    while (1) {
        CNT_INIT(run_cnt, 0);

        /* Execute one work unit from the scheduler's pool */
        /* The pool with lower index has higher priority. */
        for (i = 0; i < num_pools; i++) {
            ABT_pool pool = pools[i];
            ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
            ABT_unit unit = ABTI_pool_pop(p_pool);
            if (unit != ABT_UNIT_NULL) {
                ABTI_xstream_run_unit(&p_local_xstream, unit, p_pool);
                CNT_INC(run_cnt);
                break;
            }
        }

        if (++work_count >= event_freq) {
            ABTI_xstream_check_events(p_local_xstream, p_sched);
            ABTI_local *p_local = ABTI_xstream_get_local(p_local_xstream);
            if (ABTI_sched_has_to_stop(&p_local, p_sched) == ABT_TRUE)
                break;
            p_local_xstream = ABTI_local_get_xstream(p_local);
            work_count = 0;
            SCHED_SLEEP(run_cnt, p_data->sleep_time);
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
