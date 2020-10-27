/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 4

static ABT_barrier barrier = ABT_BARRIER_NULL;
static int num_xstreams = DEFAULT_NUM_XSTREAMS;

static void print_cpuset(int rank, int cpuset_size, int *cpuset)
{
    char *cpuset_str = NULL;
    int i;
    size_t pos = 0, len;

    cpuset_str = (char *)calloc(cpuset_size * 5, sizeof(char));
    assert(cpuset_str);

    for (i = 0; i < cpuset_size; i++) {
        if (i) {
            sprintf(&cpuset_str[pos], ",%d", cpuset[i]);
        } else {
            sprintf(&cpuset_str[pos], "%d", cpuset[i]);
        }
        len = strlen(&cpuset_str[pos]);
        pos += len;
    }
    ATS_printf(1, "[E%d] CPU set (%d): {%s}\n", rank, cpuset_size, cpuset_str);

    free(cpuset_str);
}

static void test_affinity(void *arg)
{
    ABT_xstream xstream;
    int i, rank, ret;
    int cpuid = -1, new_cpuid;
    int cpuset_size, num_cpus = -1;
    int *cpuset = NULL;

    ret = ABT_xstream_self(&xstream);
    ATS_ERROR(ret, "ABT_xstream_self");

    ret = ABT_xstream_get_rank(xstream, &rank);
    ATS_ERROR(ret, "ABT_xstream_get_rank");

    ret = ABT_xstream_get_cpubind(xstream, &cpuid);
    ATS_ERROR(ret, "ABT_xstream_get_cpubind");
    ATS_printf(1, "[E%d] CPU bind: %d\n", rank, cpuid);

    new_cpuid = (cpuid + 1) % num_xstreams;
    ATS_printf(1, "[E%d] change binding: %d -> %d\n", rank, cpuid, new_cpuid);
    ret = ABT_xstream_set_cpubind(xstream, new_cpuid);
    ATS_ERROR(ret, "ABT_xstream_set_cpubind");
    ret = ABT_xstream_get_cpubind(xstream, &cpuid);
    ATS_ERROR(ret, "ABT_xstream_get_cpubind");
    assert(cpuid == new_cpuid);
    ATS_printf(1, "[E%d] CPU bind: %d\n", rank, cpuid);

    ret = ABT_xstream_get_affinity(xstream, 0, NULL, &num_cpus);
    ATS_ERROR(ret, "ABT_xstream_get_affinity");
    ATS_printf(1, "[E%d] num_cpus=%d\n", rank, num_cpus);
    if (num_cpus > 0) {
        cpuset_size = num_cpus;
        cpuset = (int *)malloc(cpuset_size * sizeof(int));
        assert(cpuset);

        num_cpus = 0;
        ret = ABT_xstream_get_affinity(xstream, cpuset_size, cpuset, &num_cpus);
        ATS_ERROR(ret, "ABT_xstream_get_affinity");
        assert(num_cpus == cpuset_size);
        print_cpuset(rank, cpuset_size, cpuset);

        free(cpuset);
    }

    cpuset = (int *)malloc(num_xstreams * sizeof(int));
    assert(cpuset);
    for (i = 0; i < num_xstreams; i++) {
        cpuset[i] = i;
    }
    ret = ABT_xstream_set_affinity(xstream, num_xstreams, cpuset);
    ATS_ERROR(ret, "ABT_xstream_set_affinity");
    ret = ABT_xstream_get_affinity(xstream, num_xstreams, cpuset, &num_cpus);
    ATS_ERROR(ret, "ABT_xstream_get_affinity");
    assert(num_cpus == num_xstreams);
    print_cpuset(rank, num_xstreams, cpuset);

    free(cpuset);
}

int main(int argc, char *argv[])
{
    int i, ret;
    ABT_xstream primary_es;
    ABT_xstream *xstreams;
    ABT_pool *pools;
    ABT_thread *threads;
    int cpuid = -1;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc >= 2) {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "# of ESs: %d\n", num_xstreams);

    ABT_bool is_affinity_enabled = ABT_FALSE;
    ret = ABT_info_query_config(ABT_INFO_QUERY_KIND_ENABLED_AFFINITY,
                                (void *)&is_affinity_enabled);
    ATS_ERROR(ret, "ABT_info_query_config");
    if (is_affinity_enabled == ABT_FALSE) {
        /* This test should be skipped. */
        ATS_ERROR(ABT_ERR_FEATURE_NA, "ABT_info_query_config");
    }

    /* Check the affinity of the primary ES */
    ret = ABT_xstream_self(&primary_es);
    ATS_ERROR(ret, "ABT_xstream_self");
    ret = ABT_xstream_get_cpubind(primary_es, &cpuid);
    ATS_ERROR(ret, "ABT_xstream_get_cpubind");

    /* Create a barrier */
    ret = ABT_barrier_create(num_xstreams, &barrier);
    ATS_ERROR(ret, "ABT_barrier_create");

    xstreams = (ABT_xstream *)malloc(num_xstreams * sizeof(ABT_xstream));
    pools = (ABT_pool *)malloc(num_xstreams * sizeof(ABT_pool));
    threads = (ABT_thread *)malloc(num_xstreams * sizeof(ABT_thread));

    /* Create Execution Streams */
    xstreams[0] = primary_es;
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create");
    }

    /* Get the first pool of each ES */
    for (i = 0; i < num_xstreams; i++) {
        ret = ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
        ATS_ERROR(ret, "ABT_xstream_get_main_pools");
    }

    /* Create one ULT for each ES */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_thread_create(pools[i], test_affinity, (void *)(intptr_t)i,
                                ABT_THREAD_ATTR_NULL, &threads[i]);
        ATS_ERROR(ret, "ABT_thread_create");
    }

    test_affinity((void *)0);

    /* Join and free ULTs */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_thread_free(&threads[i]);
        ATS_ERROR(ret, "ABT_thread_free");
    }

    /* Join and free ESs */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Free the barrier */
    ret = ABT_barrier_free(&barrier);
    ATS_ERROR(ret, "ABT_barrier_free");

    /* Finalize */
    ret = ATS_finalize(0);

    free(xstreams);
    free(pools);
    free(threads);

    return ret;
}
