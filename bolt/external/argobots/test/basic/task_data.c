/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_NUM_TASKS 4
#define NUM_TLS 128

static ABT_key tls[NUM_TLS];
static int num_tasks;

/* tls[0] and tls[1] will be freed by the below destructor. */
static void tls_destructor(void *value)
{
    ATS_printf(1, "tls_destructor: free(%p)\n", value);
    free(value);
}

static void task_tls_test(void *arg)
{
    int my_id = (int)(intptr_t)arg;
    int i, ret;
    void *check;
    void *value[NUM_TLS];

    ATS_printf(1, "[T%d] start\n", my_id);

    /* If we haven't set a value for a key, we should get NULL for the key. */
    for (i = 0; i < NUM_TLS; i++) {
        ret = ABT_key_get(tls[i], &check);
        ATS_ERROR(ret, "ABT_key_get");
        assert(check == NULL);
    }

    for (i = 0; i < NUM_TLS; i++) {
        value[i] = malloc(16);
        ATS_printf(1, "[T%d] malloc(%p)\n", my_id, value[i]);
        ret = ABT_key_set(tls[i], value[i]);
        ATS_ERROR(ret, "ABT_key_set");
    }

    for (i = 0; i < NUM_TLS; i++) {
        ret = ABT_key_get(tls[i], &check);
        ATS_ERROR(ret, "ABT_key_get");
        assert(value[i] == check);
    }
    ATS_printf(1, "[T%d] passed #1\n", my_id);

    for (i = 0; i < NUM_TLS; i++) {
        ret = ABT_key_get(tls[i], &check);
        ATS_ERROR(ret, "ABT_key_get");
        assert(value[i] == check);
    }
    ATS_printf(1, "[T%d] passed #2\n", my_id);

    for (i = 2; i < NUM_TLS; i++) {
        ATS_printf(1, "[T%d] free(%p)\n", my_id, value[i]);
        free(value[i]);
        ret = ABT_key_set(tls[i], NULL);
        ATS_ERROR(ret, "ABT_key_set");

        ret = ABT_key_get(tls[i], &check);
        ATS_ERROR(ret, "ABT_key_get");
        assert(check == NULL);
    }
    ATS_printf(1, "[T%d] passed #3\n", my_id);

    ATS_printf(1, "[T%d] end\n", my_id);
}

static void task_create(void *arg)
{
    int i, ret;
    int my_id = (int)(intptr_t)arg;
    ABT_thread my_thread;
    ABT_pool my_pool;
    ABT_task *tasks;

    ret = ABT_thread_self(&my_thread);
    ATS_ERROR(ret, "ABT_thread_self");
    ret = ABT_thread_get_last_pool(my_thread, &my_pool);
    ATS_ERROR(ret, "ABT_thread_get_last_pool");

    /* Create tasklets */
    tasks = (ABT_task *)malloc(num_tasks * sizeof(ABT_task));
    for (i = 0; i < num_tasks; i++) {
        size_t tid = 100 * my_id + i;
        ret = ABT_task_create(my_pool, task_tls_test, (void *)tid, &tasks[i]);
        ATS_ERROR(ret, "ABT_task_create");
    }

    ATS_printf(1, "[U%d] created %d tasks\n", my_id, num_tasks);

    for (i = 0; i < num_tasks; i++) {
        ret = ABT_task_free(&tasks[i]);
        ATS_ERROR(ret, "ABT_task_free");
    }
    free(tasks);
}

int main(int argc, char *argv[])
{
    int num_xstreams;
    ABT_xstream *xstreams;
    ABT_thread *threads;
    ABT_pool *pools;
    int i, ret;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc < 2) {
        num_xstreams = DEFAULT_NUM_XSTREAMS;
        num_tasks = DEFAULT_NUM_TASKS;
    } else {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
        num_tasks = ATS_get_arg_val(ATS_ARG_N_ULT);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "# of ESs        : %d\n", num_xstreams);
    ATS_printf(1, "# of tasklets/ES: %d\n", num_tasks);
    for (i = 0; i < NUM_TLS; i++) {
        tls[i] = ABT_KEY_NULL;
    }

    xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    threads = (ABT_thread *)malloc(num_xstreams * sizeof(ABT_thread));

    /* Create WU-specific data keys */
    assert(NUM_TLS >= 4);
    for (i = 0; i < 3; i++) {
        ret = ABT_key_create(tls_destructor, &tls[i]);
        ATS_ERROR(ret, "ABT_key_create");
    }
    for (i = 3; i < NUM_TLS; i++) {
        ret = ABT_key_create(NULL, &tls[i]);
        ATS_ERROR(ret, "ABT_key_create");
    }

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the pools attached to each ES */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create one ULT for each ES */
    for (i = 1; i < num_xstreams; i++) {
        size_t tid = i + 1;
        ret = ABT_thread_create(pools[i], task_create, (void *)tid,
                                ABT_THREAD_ATTR_NULL, &threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    task_create((void *)1);

    /* Join ULTs */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_thread_free(&threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
        assert(threads[i] == ABT_THREAD_NULL);
    }

    /* Join and free Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
        assert(xstreams[i] == ABT_XSTREAM_NULL);
    }

    /* Detete keys */
    for (i = 0; i < NUM_TLS; i++) {
        ret = ABT_key_free(&tls[i]);
        ATS_ERROR(ret, "ABT_key_free");
        assert(tls[i] == ABT_KEY_NULL);
    }

    /* Finalize */
    ret = ATS_finalize(0);

    free(xstreams);
    free(pools);
    free(threads);

    return ret;
}
