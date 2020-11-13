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
#define ABTX_prof_summary(my_es, nults, iter, crea_time, crea_timestd,         \
                          crea_llcm, crea_llcmstd, crea_tlbm, crea_tlbmstd,    \
                          join_time, join_timestd, join_llcm, join_llcmstd,    \
                          join_tlbm, join_tlbmstd, free_time, free_timestd,    \
                          free_llcm, free_llcmstd, free_tlbm, free_tlbmstd)    \
    do {                                                                       \
        crea_time /= iter;                                                     \
        crea_timestd = sqrt(crea_timestd / iter - crea_time * crea_time);      \
        join_time /= iter;                                                     \
        join_timestd = sqrt(join_timestd / iter - join_time * join_time);      \
        free_time /= iter;                                                     \
        free_timestd = sqrt(free_timestd / iter - free_time * free_time);      \
        printf("%-3d %8d %8d %12.2f [%.2f] %10.2f [%.2f] %10.2f [%.2f]\n",     \
               my_es, nults, iter, crea_time, crea_timestd, join_time,         \
               join_timestd, free_time, free_timestd);                         \
        fflush(stdout);                                                        \
    } while (0)
#else
#define ABTX_prof_summary(my_es, nults, iter, crea_time, crea_timestd,         \
                          crea_llcm, crea_llcmstd, crea_tlbm, crea_tlbmstd,    \
                          join_time, join_timestd, join_llcm, join_llcmstd,    \
                          join_tlbm, join_tlbmstd, free_time, free_timestd,    \
                          free_llcm, free_llcmstd, free_tlbm, free_tlbmstd)    \
    do {                                                                       \
        crea_time /= iter;                                                     \
        crea_timestd = sqrt(crea_timestd / iter - crea_time * crea_time);      \
        crea_llcm /= iter;                                                     \
        crea_llcmstd = sqrt(crea_llcmstd / iter - crea_llcm * crea_llcm);      \
        crea_tlbm /= iter;                                                     \
        crea_tlbmstd = sqrt(crea_tlbmstd / iter - crea_tlbm * crea_tlbm);      \
        join_time /= iter;                                                     \
        join_timestd = sqrt(join_timestd / iter - join_time * join_time);      \
        join_llcm /= iter;                                                     \
        join_llcmstd = sqrt(join_llcmstd / iter - join_llcm * join_llcm);      \
        join_tlbm /= iter;                                                     \
        join_tlbmstd = sqrt(join_tlbmstd / iter - join_tlbm * join_tlbm);      \
        free_time /= iter;                                                     \
        free_timestd = sqrt(free_timestd / iter - free_time * free_time);      \
        free_llcm /= iter;                                                     \
        free_llcmstd = sqrt(free_llcmstd / iter - free_llcm * free_llcm);      \
        free_tlbm /= iter;                                                     \
        free_tlbmstd = sqrt(free_tlbmstd / iter - free_tlbm * free_tlbm);      \
        printf("%-3d %8d %8d %12.2f [%.2f] %6.2f [%.2f] %6.2f [%.2f] "         \
               "%10.2f [%.2f] %6.2f [%.2f] %6.2f [%.2f] "                      \
               "%10.2f [%.2f] %6.2f [%.2f] %6.2f [%.2f]\n",                    \
               my_es, nults, iter, crea_time, crea_timestd, crea_llcm,         \
               crea_llcmstd, crea_tlbm, crea_tlbmstd, join_time, join_timestd, \
               join_llcm, join_llcmstd, join_tlbm, join_tlbmstd, free_time,    \
               free_timestd, free_llcm, free_llcmstd, free_tlbm,               \
               free_tlbmstd);                                                  \
        fflush(stdout);                                                        \
    } while (0)
#endif

/* Initial test config in terms of #ULTs*/
#define START_NULTS 64

static ABT_xstream *xstreams;
static ABT_pool *pools;
static int niter, max_ults, ness;
static ABT_xstream_barrier g_xbarrier = ABT_XSTREAM_BARRIER_NULL;

static void thread_func(void *arg)
{
    ATS_UNUSED(arg);
}

static void main_thread_func(void *arg)
{
    int my_es = (int)(size_t)arg;
    int t, nults;

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

    ABT_thread *my_ults = (ABT_thread *)malloc(max_ults * sizeof(ABT_thread));

    /* warm-up */
    for (t = 0; t < max_ults; t++)
        ABT_thread_create(pools[my_es], thread_func, NULL, ABT_THREAD_ATTR_NULL,
                          &my_ults[t]);
#ifdef USE_JOIN_MANY
    ABT_thread_join_many(max_ults, my_ults);
#else
    for (t = 0; t < max_ults; t++)
        ABT_thread_join(my_ults[t]);
#endif
    for (t = 0; t < max_ults; t++)
        ABT_thread_free(&my_ults[t]);

    seq_state_t state;
    seq_init(&state, 2, START_NULTS / 2, START_NULTS / 2, 1);
    while ((nults = seq_get_next_term(&state)) <= max_ults) {
        ABT_xstream_barrier_wait(g_xbarrier);
        float crea_time = 0.0, crea_timestd = 0.0;
        float join_time = 0.0, join_timestd = 0.0;
        float free_time = 0.0, free_timestd = 0.0;
#ifdef USE_PAPI
        float crea_llcm = 0.0, crea_llcmstd = 0.0;
        float crea_tlbm = 0.0, crea_tlbmstd = 0.0;
        float join_llcm = 0.0, join_llcmstd = 0.0;
        float join_tlbm = 0.0, join_tlbmstd = 0.0;
        float free_llcm = 0.0, free_llcmstd = 0.0;
        float free_tlbm = 0.0, free_tlbmstd = 0.0;
#endif
        int i;

        /* The following line tries to keep the total number of iterations
         * constant */
        int iter = niter / (nults / START_NULTS);
        for (i = 0; i < iter; i++) {
            unsigned long long start_time;

            ABTX_start_prof(start_time, event_set);
            for (t = 0; t < nults; t++)
                ABT_thread_create(pools[my_es], thread_func, NULL,
                                  ABT_THREAD_ATTR_NULL, &my_ults[t]);
            ABTX_stop_prof(start_time, nults, crea_time, crea_timestd,
                           event_set, values, crea_llcm, crea_llcmstd,
                           crea_tlbm, crea_tlbmstd);

            ABTX_start_prof(start_time, event_set);
#ifdef USE_JOIN_MANY
            ABT_thread_join_many(nults, my_ults);
#else
            for (t = 0; t < nults; t++)
                ABT_thread_join(my_ults[t]);
#endif
            ABTX_stop_prof(start_time, nults, join_time, join_timestd,
                           event_set, values, join_llcm, join_llcmstd,
                           join_tlbm, join_tlbmstd);

            ABTX_start_prof(start_time, event_set);
            for (t = 0; t < nults; t++)
                ABT_thread_free(&my_ults[t]);
            ABTX_stop_prof(start_time, nults, free_time, free_timestd,
                           event_set, values, free_llcm, free_llcmstd,
                           free_tlbm, free_tlbmstd);
        }
        ABT_xstream_barrier_wait(g_xbarrier);
        ABTX_prof_summary(my_es, nults, iter, crea_time, crea_timestd,
                          crea_llcm, crea_llcmstd, crea_tlbm, crea_tlbmstd,
                          join_time, join_timestd, join_llcm, join_llcmstd,
                          join_tlbm, join_tlbmstd, free_time, free_timestd,
                          free_llcm, free_llcmstd, free_tlbm, free_tlbmstd);
    }

    free(my_ults);
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
    max_ults = ATS_get_arg_val(ATS_ARG_N_ULT);

    xstreams = (ABT_xstream *)malloc(ness * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(ness * sizeof(ABT_pool));

    ATS_init(argc, argv, ness);

    /* output beginning */
    print_header("#ULTs", 1);

    /* create a global barrier */
    ABT_xstream_barrier_create(ness, &g_xbarrier);

#ifndef USE_PRIV_POOL
    /* Create ESs*/
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
