/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_NUM_TASKS 8
#define NUM_TLS 8
#define NUM_STEPS 128

static ABT_key tls[NUM_TLS];
static int num_tasks;

static void task_f(void *arg)
{
    int i, ret;
    /* Check if there is no data race. */
    for (i = 0; i < NUM_TLS; i++) {
        void *check;
        ret = ABT_key_get(tls[i], &check);
        ATS_ERROR(ret, "ABT_key_get");
        assert(check == NULL || check == (void *)(intptr_t)i);
        ret = ABT_key_set(tls[i], (void *)(intptr_t)(i * 2));
        ATS_ERROR(ret, "ABT_key_set");
    }
}

int main(int argc, char *argv[])
{
    int num_xstreams;
    ABT_xstream *xstreams;
    ABT_task *tasks;
    ABT_pool *pools;
    int i, j, step, ret;

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

    ATS_printf(1, "# of ESs    : %d\n", num_xstreams);
    ATS_printf(1, "# of ULTs/ES: %d\n", num_tasks);
    for (i = 0; i < NUM_TLS; i++) {
        tls[i] = ABT_KEY_NULL;
    }

    xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    tasks = (ABT_task *)malloc(num_tasks * sizeof(ABT_task));

    /* Create ULT-specific data keys */
    assert(NUM_TLS >= 4);
    for (i = 0; i < NUM_TLS; i++) {
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

    for (step = 0; step < NUM_STEPS; step++) {
        /* Create one ULT for each ES */
        for (i = 1; i < num_tasks; i++) {
            ret = ABT_task_create(pools[i % num_xstreams], task_f, NULL,
                                  &tasks[i]);
            ATS_ERROR(ret, "ABT_task_create");
        }
        for (i = 1; i < num_tasks; i++) {
            for (j = 0; j < NUM_TLS; j++) {
                if (j % 2 == 0) {
                    ret = ABT_task_set_specific(tasks[i], tls[j],
                                                (void *)(intptr_t)j);
                    ATS_ERROR(ret, "ABT_task_set_specific");
                } else {
                    void *check;
                    ret = ABT_task_get_specific(tasks[i], tls[j], &check);
                    ATS_ERROR(ret, "ABT_task_get_specific");
                    assert(check == NULL || check == (void *)(intptr_t)(j * 2));
                }
            }
        }
        /* Join ULTs */
        for (i = 1; i < num_tasks; i++) {
            ret = ABT_task_free(&tasks[i]);
            ATS_ERROR(ret, "ABT_task_free");
            assert(tasks[i] == ABT_TASK_NULL);
        }
    }

    /* Join and free Execution Streams */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
        assert(xstreams[i] == ABT_XSTREAM_NULL);
    }

    /* Delete keys */
    for (i = 0; i < NUM_TLS; i++) {
        ret = ABT_key_free(&tls[i]);
        ATS_ERROR(ret, "ABT_key_free");
        assert(tls[i] == ABT_KEY_NULL);
    }

    /* Finalize */
    ret = ATS_finalize(0);

    free(xstreams);
    free(pools);
    free(tasks);

    return ret;
}
