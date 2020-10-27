/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#ifdef USE_PAPI
#include <papi.h>
#endif
#include "abt.h"
#include "abttest.h"
#include "bench_util.h"

#ifndef USE_PAPI
#define ABTX_prof_summary(my_es, ntasks, iter, crea_time, crea_timestd,        \
                          crea_llcm, crea_llcmstd, crea_tlbm, crea_tlbmstd,    \
                          free_time, free_timestd, free_llcm, free_llcmstd,    \
                          free_tlbm, free_tlbmstd)                             \
    do {                                                                       \
        crea_time /= iter;                                                     \
        crea_timestd = sqrt(crea_timestd / iter - crea_time * crea_time);      \
        free_time /= iter;                                                     \
        free_timestd = sqrt(free_timestd / iter - free_time * free_time);      \
        printf("%-3d %8d %8d %12.2f [%.2f] %10.2f [%.2f]\n", my_es, ntasks,    \
               iter, crea_time, crea_timestd, free_time, free_timestd);        \
        fflush(stdout);                                                        \
    } while (0)
#else
#define ABTX_prof_summary(my_es, ntasks, iter, crea_time, crea_timestd,        \
                          crea_llcm, crea_llcmstd, crea_tlbm, crea_tlbmstd,    \
                          free_time, free_timestd, free_llcm, free_llcmstd,    \
                          free_tlbm, free_tlbmstd)                             \
    do {                                                                       \
        crea_time /= iter;                                                     \
        crea_timestd = sqrt(crea_timestd / iter - crea_time * crea_time);      \
        crea_llcm /= iter;                                                     \
        crea_llcmstd = sqrt(crea_llcmstd / iter - crea_llcm * crea_llcm);      \
        crea_tlbm /= iter;                                                     \
        crea_tlbmstd = sqrt(crea_tlbmstd / iter - crea_tlbm * crea_tlbm);      \
        free_time /= iter;                                                     \
        free_timestd = sqrt(free_timestd / iter - free_time * free_time);      \
        free_llcm /= iter;                                                     \
        free_llcmstd = sqrt(free_llcmstd / iter - free_llcm * free_llcm);      \
        free_tlbm /= iter;                                                     \
        free_tlbmstd = sqrt(free_tlbmstd / iter - free_tlbm * free_tlbm);      \
        printf("%-3d %8d %8d %12.2f [%.2f] %6.2f [%.2f] %6.2f [%.2f] "         \
               "%10.2f [%.2f] %6.2f [%.2f] %6.2f [%.2f]\n",                    \
               my_es, ntasks, iter, crea_time, crea_timestd, crea_llcm,        \
               crea_llcmstd, crea_tlbm, crea_tlbmstd, free_time, free_timestd, \
               free_llcm, free_llcmstd, free_tlbm, free_tlbmstd);              \
        fflush(stdout);                                                        \
    } while (0)
#endif

/* Initial test config in terms of #Tasks*/
#define START_NTASKS 64

static ABT_xstream *xstreams;
static ABT_pool *pools;
static int niter, max_tasks, ness;
static ABT_xstream_barrier g_xbarrier = ABT_XSTREAM_BARRIER_NULL;

static void task_func(void *arg)
{
    ATS_UNUSED(arg);
}

static void main_thread_func(void *arg)
{
    int my_es = (int)(size_t)arg;
    int t, ntasks;

#ifdef USE_PAPI
    /* Create the Event Set */
    int event_set = PAPI_NULL;
    long_long values[2];
    if (my_es > 0) {
        ABTX_papi_assert(PAPI_register_thread());
    }
    ABTX_papi_assert(PAPI_create_eventset(&event_set));

    /* Add events to monitor */
    ABTX_papi_add_event(event_set);
#endif /* USE_PAPI */

    ABT_task *my_tasks = (ABT_task *)malloc(max_tasks * sizeof(ABT_task));

    /* warm-up */
    for (t = 0; t < max_tasks; t++)
        ABT_task_create(pools[my_es], task_func, NULL, &my_tasks[t]);
    for (t = 0; t < max_tasks; t++)
        ABT_task_free(&my_tasks[t]);

    seq_state_t state;
    seq_init(&state, 2, START_NTASKS / 2, START_NTASKS / 2, 1);
    while ((ntasks = seq_get_next_term(&state)) <= max_tasks) {
        ABT_xstream_barrier_wait(g_xbarrier);
        float crea_time = 0.0, crea_timestd = 0.0;
        float free_time = 0.0, free_timestd = 0.0;
#ifdef USE_PAPI
        float crea_llcm = 0.0, crea_llcmstd = 0.0;
        float crea_tlbm = 0.0, crea_tlbmstd = 0.0;
        float free_llcm = 0.0, free_llcmstd = 0.0;
        float free_tlbm = 0.0, free_tlbmstd = 0.0;
#endif
        int i;

        /* The following line tries to keep the total number of iterations
         * constant */
        int iter = niter / (ntasks / START_NTASKS);
        for (i = 0; i < iter; i++) {
            unsigned long long start_time;

            ABTX_start_prof(start_time, event_set);
            for (t = 0; t < ntasks; t++)
                ABT_task_create(pools[my_es], task_func, NULL, &my_tasks[t]);
            ABTX_stop_prof(start_time, ntasks, crea_time, crea_timestd,
                           event_set, values, crea_llcm, crea_llcmstd,
                           crea_tlbm, crea_tlbmstd);

            ABTX_start_prof(start_time, event_set);
            for (t = 0; t < ntasks; t++)
                ABT_task_free(&my_tasks[t]);
            ABTX_stop_prof(start_time, ntasks, free_time, free_timestd,
                           event_set, values, free_llcm, free_llcmstd,
                           free_tlbm, free_tlbmstd);
        }
        ABT_xstream_barrier_wait(g_xbarrier);
        ABTX_prof_summary(my_es, ntasks, iter, crea_time, crea_timestd,
                          crea_llcm, crea_llcmstd, crea_tlbm, crea_tlbmstd,
                          free_time, free_timestd, free_llcm, free_llcmstd,
                          free_tlbm, free_tlbmstd);
    }

    free(my_tasks);
#ifdef USE_PAPI
    if (my_es > 0) {
        ABTX_papi_assert(PAPI_unregister_thread());
    }
#endif
}

int main(int argc, char *argv[])
{
#ifdef USE_PAPI
    int ret = PAPI_library_init(PAPI_VER_CURRENT);
    if (ret != PAPI_VER_CURRENT) {
        fprintf(stderr, "PAPI library init error!\n");
        exit(1);
    }
    ABTX_papi_assert(PAPI_thread_init(ABTX_xstream_get_self));
#endif

    int i;
    ATS_read_args(argc, argv);
    niter = ATS_get_arg_val(ATS_ARG_N_ITER);
    ness = ATS_get_arg_val(ATS_ARG_N_ES);
    max_tasks = ATS_get_arg_val(ATS_ARG_N_TASK);

    xstreams = (ABT_xstream *)malloc(ness * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(ness * sizeof(ABT_pool));

    ATS_init(argc, argv, ness);

    /* output beginning */
    print_header("#Tasks", 0);

    /* create a global barrier */
    ABT_xstream_barrier_create(ness, &g_xbarrier);

#ifndef USE_PRIV_POOL
    /* Create ESs */
    ABT_xstream_self(&xstreams[0]);
    ABT_xstream_get_main_pools(xstreams[0], 1, &pools[0]);
    for (i = 1; i < ness; i++) {
        ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
    }

    for (i = 1; i < ness; i++) {
        ABT_thread_create(pools[i], main_thread_func, (void *)(size_t)i,
                          ABT_THREAD_ATTR_NULL, NULL);
    }

    main_thread_func((void *)(size_t)0);
#else
    /* Create pools */
    for (i = 0; i < ness; i++) {
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_PRIV, ABT_TRUE,
                              &pools[i]);
    }

    for (i = 1; i < ness; i++) {
        ABT_thread_create(pools[i], main_thread_func, (void *)(size_t)i,
                          ABT_THREAD_ATTR_NULL, NULL);
    }

    /* Create ESs*/
    ABT_xstream_self(&xstreams[0]);
    ABT_xstream_set_main_sched_basic(xstreams[0], ABT_SCHED_DEFAULT, 1,
                                     &pools[0]);
    for (i = 1; i < ness; i++) {
        ABT_xstream_create_basic(ABT_SCHED_DEFAULT, 1, &pools[i],
                                 ABT_SCHED_CONFIG_NULL, &xstreams[i]);
    }

    main_thread_func((void *)(size_t)0);
#endif

    for (i = 1; i < ness; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }
    ABT_xstream_barrier_free(&g_xbarrier);

    ATS_finalize(0);

    free(pools);
    free(xstreams);

    return EXIT_SUCCESS;
}
