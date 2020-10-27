/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

#define EVENTUAL_SIZE 10

ABT_thread th1, th2, th3;
ABT_eventual myeventual;

#define LOOP_CNT 10
void fn1(void *args)
{
    ATS_UNUSED(args);
    int i = 0;
    void *data;
    ATS_printf(1, "Thread 1 iteration %d waiting for eventual\n", i);
    ABT_eventual_wait(myeventual, &data);
    ATS_printf(1,
               "Thread 1 continue iteration %d returning from "
               "eventual\n",
               i);
}

void fn2(void *args)
{
    ATS_UNUSED(args);
    int i = 0, is_ready = 0;
    void *data;
    ATS_printf(1, "Thread 2 iteration %d waiting from eventual\n", i);
    ABT_eventual_test(myeventual, &data, &is_ready);
    while (!is_ready) {
        ABT_thread_yield();
        ABT_eventual_test(myeventual, &data, &is_ready);
    }
    ABT_eventual_wait(myeventual, &data);
    ATS_printf(1,
               "Thread 2 continue iteration %d returning from "
               "eventual\n",
               i);
}

void fn3(void *args)
{
    ATS_UNUSED(args);
    int i = 0;
    ATS_printf(1, "Thread 3 iteration %d signal eventual \n", i);
    char *data = (char *)malloc(EVENTUAL_SIZE);
    ABT_eventual_set(myeventual, data, EVENTUAL_SIZE);
    free(data);
    ATS_printf(1, "Thread 3 continue iteration %d \n", i);
}

int main(int argc, char *argv[])
{
    int ret;
    ABT_xstream xstream;

    /* init and thread creation */
    ATS_read_args(argc, argv);
    ATS_init(argc, argv, 1);

    ret = ABT_xstream_self(&xstream);
    ATS_ERROR(ret, "ABT_xstream_self");

    /* Get the pools attached to an execution stream */
    ABT_pool pool;
    ret = ABT_xstream_get_main_pools(xstream, 1, &pool);
    ATS_ERROR(ret, "ABT_xstream_get_main_pools");

    ret = ABT_thread_create(pool, fn1, NULL, ABT_THREAD_ATTR_NULL, &th1);
    ATS_ERROR(ret, "ABT_thread_create");
    ret = ABT_thread_create(pool, fn2, NULL, ABT_THREAD_ATTR_NULL, &th2);
    ATS_ERROR(ret, "ABT_thread_create");
    ret = ABT_thread_create(pool, fn3, NULL, ABT_THREAD_ATTR_NULL, &th3);
    ATS_ERROR(ret, "ABT_thread_create");

    ret = ABT_eventual_create(EVENTUAL_SIZE, &myeventual);
    ATS_ERROR(ret, "ABT_eventual_create");

    ATS_printf(1, "START\n");

    void *data;
    ATS_printf(1, "Thread main iteration %d waiting for eventual\n", 0);
    ABT_eventual_wait(myeventual, &data);
    ATS_printf(1,
               "Thread main continue iteration %d returning from "
               "eventual\n",
               0);

    /* Join and free other threads */
    ret = ABT_thread_free(&th1);
    ATS_ERROR(ret, "ABT_thread_free");
    ret = ABT_thread_free(&th2);
    ATS_ERROR(ret, "ABT_thread_free");
    ret = ABT_thread_free(&th3);
    ATS_ERROR(ret, "ABT_thread_free");

    ATS_printf(1, "END\n");

    ret = ABT_eventual_free(&myeventual);
    ATS_ERROR(ret, "ABT_eventual_free");

    ret = ATS_finalize(0);

    return ret;
}
