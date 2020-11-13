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
static int example_pool_get_def(ABT_pool_access access, ABT_pool_def *p_def);
static void create_threads(void *arg);
static void thread_hello(void *arg);

int main(int argc, char *argv[])
{
    ABT_xstream xstreams[NUM_XSTREAMS];
    ABT_sched scheds[NUM_XSTREAMS];
    ABT_pool pools[NUM_XSTREAMS];
    ABT_thread threads[NUM_XSTREAMS];
    ABT_pool_def pool_def;
    int i;

    ABT_init(argc, argv);

    /* Create pools */
    example_pool_get_def(ABT_POOL_ACCESS_MPMC, &pool_def);
    for (i = 0; i < NUM_XSTREAMS; i++) {
        ABT_pool_create(&pool_def, ABT_POOL_CONFIG_NULL, &pools[i]);
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

/* FIFO pool implementation
 *
 * Based on src/pool/fifo.c, but modified to avoid the use of internal data
 * structures.
 */

/*******************************************************************/

struct example_unit {
    struct example_unit *p_prev;
    struct example_unit *p_next;
    ABT_pool pool;
    union {
        ABT_thread thread;
        ABT_task task;
    } handle;
    ABT_unit_type type;
};

struct example_pool_data {
    ABT_mutex mutex;
    size_t num_units;
    struct example_unit *p_head;
    struct example_unit *p_tail;
};

static int pool_init(ABT_pool pool, ABT_pool_config config);
static int pool_free(ABT_pool pool);
static size_t pool_get_size(ABT_pool pool);
static void pool_push_shared(ABT_pool pool, ABT_unit unit);
static void pool_push_private(ABT_pool pool, ABT_unit unit);
static ABT_unit pool_pop_shared(ABT_pool pool);
static ABT_unit pool_pop_private(ABT_pool pool);
static int pool_remove_shared(ABT_pool pool, ABT_unit unit);
static int pool_remove_private(ABT_pool pool, ABT_unit unit);

typedef struct example_unit unit_t;
static ABT_unit_type unit_get_type(ABT_unit unit);
static ABT_thread unit_get_thread(ABT_unit unit);
static ABT_task unit_get_task(ABT_unit unit);
static ABT_bool unit_is_in_pool(ABT_unit unit);
static ABT_unit unit_create_from_thread(ABT_thread thread);
static ABT_unit unit_create_from_task(ABT_task task);
static void unit_free(ABT_unit *unit);

typedef struct example_pool_data data_t;

static inline data_t *pool_get_data_ptr(void *p_data)
{
    return (data_t *)p_data;
}

/* Obtain the FIFO pool definition according to the access type */
static int example_pool_get_def(ABT_pool_access access, ABT_pool_def *p_def)
{
    int abt_errno = ABT_SUCCESS;

    /* Definitions according to the access type */
    /* FIXME: need better implementation, e.g., lock-free one */
    switch (access) {
        case ABT_POOL_ACCESS_PRIV:
            p_def->p_push = pool_push_private;
            p_def->p_pop = pool_pop_private;
            p_def->p_remove = pool_remove_private;
            break;

        case ABT_POOL_ACCESS_SPSC:
        case ABT_POOL_ACCESS_MPSC:
        case ABT_POOL_ACCESS_SPMC:
        case ABT_POOL_ACCESS_MPMC:
            p_def->p_push = pool_push_shared;
            p_def->p_pop = pool_pop_shared;
            p_def->p_remove = pool_remove_shared;
            break;

        default:
            return ABT_ERR_INV_POOL_ACCESS;
    }

    /* Common definitions regardless of the access type */
    p_def->access = access;
    p_def->p_init = pool_init;
    p_def->p_free = pool_free;
    p_def->p_get_size = pool_get_size;
    p_def->u_get_type = unit_get_type;
    p_def->u_get_thread = unit_get_thread;
    p_def->u_get_task = unit_get_task;
    p_def->u_is_in_pool = unit_is_in_pool;
    p_def->u_create_from_thread = unit_create_from_thread;
    p_def->u_create_from_task = unit_create_from_task;
    p_def->u_free = unit_free;

    return abt_errno;
}

/* Pool functions */

int pool_init(ABT_pool pool, ABT_pool_config config)
{
    int abt_errno = ABT_SUCCESS;
    ABT_pool_access access;

    data_t *p_data = (data_t *)malloc(sizeof(data_t));
    if (!p_data)
        return ABT_ERR_MEM;

    ABT_pool_get_access(pool, &access);

    p_data->mutex = ABT_MUTEX_NULL;
    if (access != ABT_POOL_ACCESS_PRIV) {
        /* Initialize the mutex */
        ABT_mutex_create(&p_data->mutex);
    }

    p_data->num_units = 0;
    p_data->p_head = NULL;
    p_data->p_tail = NULL;

    ABT_pool_set_data(pool, p_data);

    return abt_errno;
}

static int pool_free(ABT_pool pool)
{
    int abt_errno = ABT_SUCCESS;
    void *data;
    ABT_pool_get_data(pool, &data);
    data_t *p_data = pool_get_data_ptr(data);
    if (p_data->mutex != ABT_MUTEX_NULL) {
        ABT_mutex_free(&p_data->mutex);
    }

    free(p_data);

    return abt_errno;
}

static size_t pool_get_size(ABT_pool pool)
{
    void *data;
    ABT_pool_get_data(pool, &data);
    data_t *p_data = pool_get_data_ptr(data);
    return p_data->num_units;
}

static void pool_push_shared(ABT_pool pool, ABT_unit unit)
{
    void *data;
    ABT_pool_get_data(pool, &data);
    data_t *p_data = pool_get_data_ptr(data);
    unit_t *p_unit = (unit_t *)unit;

    ABT_mutex_spinlock(p_data->mutex);
    if (p_data->num_units == 0) {
        p_unit->p_prev = p_unit;
        p_unit->p_next = p_unit;
        p_data->p_head = p_unit;
        p_data->p_tail = p_unit;
    } else {
        unit_t *p_head = p_data->p_head;
        unit_t *p_tail = p_data->p_tail;
        p_tail->p_next = p_unit;
        p_head->p_prev = p_unit;
        p_unit->p_prev = p_tail;
        p_unit->p_next = p_head;
        p_data->p_tail = p_unit;
    }
    p_data->num_units++;

    p_unit->pool = pool;
    ABT_mutex_unlock(p_data->mutex);
}

static void pool_push_private(ABT_pool pool, ABT_unit unit)
{
    void *data;
    ABT_pool_get_data(pool, &data);
    data_t *p_data = pool_get_data_ptr(data);
    unit_t *p_unit = (unit_t *)unit;

    if (p_data->num_units == 0) {
        p_unit->p_prev = p_unit;
        p_unit->p_next = p_unit;
        p_data->p_head = p_unit;
        p_data->p_tail = p_unit;
    } else {
        unit_t *p_head = p_data->p_head;
        unit_t *p_tail = p_data->p_tail;
        p_tail->p_next = p_unit;
        p_head->p_prev = p_unit;
        p_unit->p_prev = p_tail;
        p_unit->p_next = p_head;
        p_data->p_tail = p_unit;
    }
    p_data->num_units++;

    p_unit->pool = pool;
}

static ABT_unit pool_pop_shared(ABT_pool pool)
{
    void *data;
    ABT_pool_get_data(pool, &data);
    data_t *p_data = pool_get_data_ptr(data);
    unit_t *p_unit = NULL;
    ABT_unit h_unit = ABT_UNIT_NULL;

    ABT_mutex_spinlock(p_data->mutex);
    if (p_data->num_units > 0) {
        p_unit = p_data->p_head;
        if (p_data->num_units == 1) {
            p_data->p_head = NULL;
            p_data->p_tail = NULL;
        } else {
            p_unit->p_prev->p_next = p_unit->p_next;
            p_unit->p_next->p_prev = p_unit->p_prev;
            p_data->p_head = p_unit->p_next;
        }
        p_data->num_units--;

        p_unit->p_prev = NULL;
        p_unit->p_next = NULL;
        p_unit->pool = ABT_POOL_NULL;

        h_unit = (ABT_unit)p_unit;
    }
    ABT_mutex_unlock(p_data->mutex);

    return h_unit;
}

static ABT_unit pool_pop_private(ABT_pool pool)
{
    void *data;
    ABT_pool_get_data(pool, &data);
    data_t *p_data = pool_get_data_ptr(data);
    unit_t *p_unit = NULL;
    ABT_unit h_unit = ABT_UNIT_NULL;

    if (p_data->num_units > 0) {
        p_unit = p_data->p_head;
        if (p_data->num_units == 1) {
            p_data->p_head = NULL;
            p_data->p_tail = NULL;
        } else {
            p_unit->p_prev->p_next = p_unit->p_next;
            p_unit->p_next->p_prev = p_unit->p_prev;
            p_data->p_head = p_unit->p_next;
        }
        p_data->num_units--;

        p_unit->p_prev = NULL;
        p_unit->p_next = NULL;
        p_unit->pool = ABT_POOL_NULL;

        h_unit = (ABT_unit)p_unit;
    }

    return h_unit;
}

static int pool_remove_shared(ABT_pool pool, ABT_unit unit)
{
    void *data;
    ABT_pool_get_data(pool, &data);
    data_t *p_data = pool_get_data_ptr(data);
    unit_t *p_unit = (unit_t *)unit;

    if (p_data->num_units == 0)
        return ABT_ERR_POOL;
    if (p_unit->pool == ABT_POOL_NULL)
        return ABT_ERR_POOL;

    if (p_unit->pool != pool) {
        return ABT_ERR_INV_POOL;
    }

    ABT_mutex_spinlock(p_data->mutex);
    if (p_data->num_units == 1) {
        p_data->p_head = NULL;
        p_data->p_tail = NULL;
    } else {
        p_unit->p_prev->p_next = p_unit->p_next;
        p_unit->p_next->p_prev = p_unit->p_prev;
        if (p_unit == p_data->p_head) {
            p_data->p_head = p_unit->p_next;
        } else if (p_unit == p_data->p_tail) {
            p_data->p_tail = p_unit->p_prev;
        }
    }
    p_data->num_units--;

    p_unit->pool = ABT_POOL_NULL;
    ABT_mutex_unlock(p_data->mutex);

    p_unit->p_prev = NULL;
    p_unit->p_next = NULL;

    return ABT_SUCCESS;
}

static int pool_remove_private(ABT_pool pool, ABT_unit unit)
{
    void *data;
    ABT_pool_get_data(pool, &data);
    data_t *p_data = pool_get_data_ptr(data);
    unit_t *p_unit = (unit_t *)unit;

    if (p_data->num_units == 0)
        return ABT_ERR_POOL;
    if (p_unit->pool == ABT_POOL_NULL)
        return ABT_ERR_POOL;

    if (p_unit->pool != pool) {
        return ABT_ERR_INV_POOL;
    }

    if (p_data->num_units == 1) {
        p_data->p_head = NULL;
        p_data->p_tail = NULL;
    } else {
        p_unit->p_prev->p_next = p_unit->p_next;
        p_unit->p_next->p_prev = p_unit->p_prev;
        if (p_unit == p_data->p_head) {
            p_data->p_head = p_unit->p_next;
        } else if (p_unit == p_data->p_tail) {
            p_data->p_tail = p_unit->p_prev;
        }
    }
    p_data->num_units--;

    p_unit->pool = ABT_POOL_NULL;
    p_unit->p_prev = NULL;
    p_unit->p_next = NULL;

    return ABT_SUCCESS;
}

/* Unit functions */

static ABT_unit_type unit_get_type(ABT_unit unit)
{
    unit_t *p_unit = (unit_t *)unit;
    return p_unit->type;
}

static ABT_thread unit_get_thread(ABT_unit unit)
{
    ABT_thread h_thread;
    unit_t *p_unit = (unit_t *)unit;
    if (p_unit->type == ABT_UNIT_TYPE_THREAD) {
        h_thread = p_unit->handle.thread;
    } else {
        h_thread = ABT_THREAD_NULL;
    }
    return h_thread;
}

static ABT_task unit_get_task(ABT_unit unit)
{
    ABT_task h_task;
    unit_t *p_unit = (unit_t *)unit;
    if (p_unit->type == ABT_UNIT_TYPE_TASK) {
        h_task = p_unit->handle.task;
    } else {
        h_task = ABT_TASK_NULL;
    }
    return h_task;
}

static ABT_bool unit_is_in_pool(ABT_unit unit)
{
    unit_t *p_unit = (unit_t *)unit;
    return (p_unit->pool != ABT_POOL_NULL) ? ABT_TRUE : ABT_FALSE;
}

static ABT_unit unit_create_from_thread(ABT_thread thread)
{
    unit_t *p_unit = malloc(sizeof(unit_t));
    if (!p_unit)
        return ABT_UNIT_NULL;

    p_unit->p_prev = NULL;
    p_unit->p_next = NULL;
    p_unit->pool = ABT_POOL_NULL;
    p_unit->handle.thread = thread;
    p_unit->type = ABT_UNIT_TYPE_THREAD;

    return (ABT_unit)p_unit;
}

static ABT_unit unit_create_from_task(ABT_task task)
{
    unit_t *p_unit = malloc(sizeof(unit_t));
    if (!p_unit)
        return ABT_UNIT_NULL;

    p_unit->p_prev = NULL;
    p_unit->p_next = NULL;
    p_unit->pool = ABT_POOL_NULL;
    p_unit->handle.task = task;
    p_unit->type = ABT_UNIT_TYPE_TASK;

    return (ABT_unit)p_unit;
}

static void unit_free(ABT_unit *unit)
{
    free(*unit);
    *unit = ABT_UNIT_NULL;
}
