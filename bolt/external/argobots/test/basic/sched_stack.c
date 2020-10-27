/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/* In this test, we use only one ES. We create one main pool (attached to the
 * main scheduler). First we push one task in this pool, then one
 * sub-scheduler, then another task, then another sub-scheduler and finally one
 * last task. In the tasks, we check the order of execution.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_TASKS 8

long int value = 0;

void task_func(void *arg)
{
    long int v = (long int)arg;
    assert(v == value + 1);
    value++;
}

int main(int argc, char *argv[])
{

    int i, ret;
    int num_tasks = DEFAULT_NUM_TASKS;

    if (argc > 1)
        num_tasks = atoi(argv[1]);
    assert(num_tasks >= 0);

    ABT_xstream xstream;

    ABT_pool pool_mainsched, pool_subsched1, pool_subsched2;
    ABT_sched mainsched, subsched1, subsched2;

    /* Initialize */
    ATS_read_args(argc, argv);
    ATS_init(argc, argv, 1);

    /* Creation of the main pool/sched */
    ret = ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_PRIV, ABT_TRUE,
                                &pool_mainsched);
    ATS_ERROR(ret, "ABT_pool_create_basic");
    ret = ABT_sched_create_basic(ABT_SCHED_DEFAULT, 1, &pool_mainsched,
                                 ABT_SCHED_CONFIG_NULL, &mainsched);
    ATS_ERROR(ret, "ABT_sched_create_basic");

    /* Configure the main Execution Stream with mainsched */
    ret = ABT_xstream_self(&xstream);
    ATS_ERROR(ret, "ABT_xstream_self");
    ret = ABT_xstream_set_main_sched(xstream, mainsched);
    ATS_ERROR(ret, "ABT_xstream_set_main_sched");

    /* Creation of subsched1 */
    ret = ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_PRIV, ABT_TRUE,
                                &pool_subsched1);
    ATS_ERROR(ret, "ABT_pool_create_basic");
    ret = ABT_sched_create_basic(ABT_SCHED_DEFAULT, 1, &pool_subsched1,
                                 ABT_SCHED_CONFIG_NULL, &subsched1);
    ATS_ERROR(ret, "ABT_sched_create_basic");

    /* Creation of subsched2 */
    ret = ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_PRIV, ABT_TRUE,
                                &pool_subsched2);
    ATS_ERROR(ret, "ABT_pool_create_basic");
    ret = ABT_sched_create_basic(ABT_SCHED_DEFAULT, 1, &pool_subsched2,
                                 ABT_SCHED_CONFIG_NULL, &subsched2);
    ATS_ERROR(ret, "ABT_sched_create_basic");

    long int num = 0;
    ret = ABT_task_create(pool_mainsched, task_func, (void *)++num, NULL);
    ATS_ERROR(ret, "ABT_task_create");

    for (i = 0; i < num_tasks; i++) {
        ret = ABT_task_create(pool_subsched1, task_func, (void *)++num, NULL);
        ATS_ERROR(ret, "ABT_task_create");
    }
    ret = ABT_pool_add_sched(pool_mainsched, subsched1);
    ATS_ERROR(ret, "ABT_pool_add_sched");

    ret = ABT_task_create(pool_mainsched, task_func, (void *)++num, NULL);
    ATS_ERROR(ret, "ABT_task_create");

    for (i = 0; i < num_tasks; i++) {
        ret = ABT_task_create(pool_subsched2, task_func, (void *)++num, NULL);
        ATS_ERROR(ret, "ABT_task_create");
    }
    ret = ABT_pool_add_sched(pool_mainsched, subsched2);
    ATS_ERROR(ret, "ABT_pool_add_sched");

    ret = ABT_task_create(pool_mainsched, task_func, (void *)++num, NULL);
    ATS_ERROR(ret, "ABT_task_create");

    /* Finalize */
    ret = ATS_finalize(0);

    assert(value == 3 + 2 * num_tasks);
    return ret;
}
