/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * The Argobots affinity module parses the following grammar while it ignores
 * all white spaces between tokens.  Double-quoted symbols in the following are
 * part of the syntax.
 *
 * <list>         = <interval>
 *                | <list> "," <interval>
 * <interval>     = <es-id-list> ":" <num> ":" <stride>
 *                | <es-id-list> ":" <num>
 *                | <es-id-list>
 * <es-id-list>   = <id>
 *                | "{" <id-list> "}"
 * <id-list>      = <id-interval>
 *                | <id-list> "," <id-interval>
 * <id-interval>  = <id> ":" <num> ":" <stride>
 *                | <id> ":" <num>
 *                | <id>
 * <id>           = <integer>
 * <stride>       = <integer>
 * <num>          = <positive integer>
 *
 * An execution stream with rank n refers to the (n % N)th CPU ID list
 * (<es-id-list>) of the whole list (<list>) that has N items.  The execution
 * stream will be scheduled on a core that has a CPU ID in its CPU ID list
 * (<es-id-list>).  If a CPU ID is smaller than zero or larger than the number
 * of cores recognized by the system, a modulo of the number of cores is used.
 *
 * This grammar supports a pattern-based syntax.  <id-interval> can represent
 * multiple CPU IDs by specifying a base CPU ID (<id>), the number of CPU IDs
 * (<num>), and a stride (<stride>) as follows:
 *   <id>, <id> + <stride>, ..., <id> + <stride> * (<num> - 1)
 * <interval> also accepts a pattern-based syntax as follows:
 *   <es-id-list>,
 *   {<es-id-list>[0] + <stride>, <es-id-list>[1] + <stride>, ...}, ...
 *   {<es-id-list>[0] + <stride> * (<num> - 1),
 *    <es-id-list>[1] + <stride> * (<num> - 1), ...}
 * Note that <num> and <stride> are set to 1 if they are omitted.
 *
 * Let us assume a 12-core machine (NOTE: "12-core" does not mean 12 physical
 * cores but indicates 12 hardware threads on modern CPUs).  Examples are as
 * follows:
 *
 * Example 1: bind ES with rank n to a core that has CPU ID = n % 12
 *
 * | ES0 | ES1 | ES2 | ES3 | ES4 | ES5 | ES6 | ES7 | ES8 | ES9 | ES10| ES11|
 * |CPU00|CPU01|CPU02|CPU03|CPU04|CPU05|CPU06|CPU07|CPU08|CPU09|CPU10|CPU11|
 *
 * (1.1) ABT_SET_AFFINITY="{0},{1},{2},{3},{4},{5},{6},{7},{8},{9},{10},{11}"
 *      This explicitly sets all the CPU IDs.
 * (1.2) ABT_SET_AFFINITY="0,1,2,3,4,5,6,7,8,9,10,11"
 *      It is similar to (1.1), but it omits "{}" since "{id}" = "id".
 * (1.3) ABT_SET_AFFINITY="{0}:12:1"
 *      This creates 12 CPU ID lists starting from {0}.  The stride is specified
 *      as 1, so the created lists are {0}, {1}, {2}, ... {11}.
 * (1.4) ABT_SET_AFFINITY="{0}:12"
 *      The default stride is 1, so it is the same as (1.3)
 * (1.5) ABT_SET_AFFINITY="0:12"
 *      It omits "{}" in (1.4) since "{id}" = "id".
 *
 * Example 2: bind ES with rank n to a core that has CPU ID = (n + 6) % 12
 *
 * | ES6 | ES7 | ES8 | ES9 | ES10| ES11| ES0 | ES1 | ES2 | ES3 | ES4 | ES5 |
 * |CPU00|CPU01|CPU02|CPU03|CPU04|CPU05|CPU06|CPU07|CPU08|CPU09|CPU10|CPU11|
 *
 * (2.1) ABT_SET_AFFINITY="{6},{7},{8},{9},{10},{11},{0},{1},{2},{3},{4},{5}"
 *      This explicitly sets all the CPU IDs.
 * (2.2) ABT_SET_AFFINITY="6,7,8,9,10,11,0,1,2,3,4,5"
 *      It is similar to (2.1), but it omits "{}" since "{id}" = "id".
 * (2.3) ABT_SET_AFFINITY="{6}:6:1,{0}:6:1"
 *      This first creates 6 CPU ID lists starting from {6} and then 6 lists
 *      starting from {0}.  The stride is "1", so the created lists are
 *      {6}, ... {11}, {0} ... {5}.
 * (2.4) ABT_SET_AFFINITY="{6}:6,{0}:6"
 *      The default stride is 1, so it is the same as (2.3)
 * (2.5) ABT_SET_AFFINITY="6:6,0:6"
 *      It omits "{}" in (2.4) since "{id}" = "id".
 * (2.6) ABT_SET_AFFINITY="6:12"
 *      Affinity setting wraps around with respect to the number of cores, so
 *      this works the same as (2.5) on a 12-core machine.
 *
 * Example 3: bind ES with rank n to core that has CPU ID = (n * 4) % 12
 *
 * | ES0 |     |     |     | ES1 |     |     |     | ES2 |     |     |     |
 * |CPU00|CPU01|CPU02|CPU03|CPU04|CPU05|CPU06|CPU07|CPU08|CPU09|CPU10|CPU11|
 *
 * (3.1) ABT_SET_AFFINITY="{0},{4},{8}"
 *      This explicitly sets all the CPU IDs.
 * (3.2) ABT_SET_AFFINITY="0,4,8"
 *      It is similar to (3.1), but it omits "{}" since "{id}" = "id".
 * (3.3) ABT_SET_AFFINITY="{0}:3:4"
 *      This creates 3 CPU ID lists starting from {0}. The stride is "4", so
 *      creates lists are {0}, {4}, {8}.
 * (3.4) ABT_SET_AFFINITY="0:3:4"
 *      It omits "{}" in (3.3) since "{id}" = "id".
 *
 * Example 4: bind ES with rank n to core that has
 *            CPU ID = (n * 4 + (n % 12) / 3) % 12
 *
 * | ES0 | ES3 | ES6 | ES9 | ES1 | ES4 | ES7 | ES10| ES2 | ES5 | ES9 | ES11|
 * |CPU00|CPU01|CPU02|CPU03|CPU04|CPU05|CPU06|CPU07|CPU08|CPU09|CPU10|CPU11|
 *
 * (4.1) ABT_SET_AFFINITY="{0},{4},{8},{1},{5},{9},{2},{6},{10},{3},{7},{11}"
 *      This explicitly sets all the CPU IDs.
 * (4.2) ABT_SET_AFFINITY="0,4,8,1,5,9,2,6,10,3,7,11"
 *      It is similar to (4.1), but it omits "{}" since "{id}" = "id".
 * (4.3) ABT_SET_AFFINITY="{0}:3:4,{1}:3:4,{2}:3:4,{3}:3:4"
 *      This creates 3 CPU ID lists ({0}, {4}, {8}), then ({1}, {5}, {9}),
 *      ({2}, {6}, {10}), and ({3}, {7}, {11}).
 * (4.4) ABT_SET_AFFINITY="0:3:4,1:3:4,2:3:4,3:3:4"
 *      It omits "{}" in (4.3) since "{id}" = "id".
 *
 * Example 5: bind ES with rank n to cores that have
 *            (n * 4) % 12 <= CPU ID < (n * 4) % 12 + 4
 *
 * |          ES0          |          ES1          |          ES2          |
 * |CPU00|CPU01|CPU02|CPU03|CPU04|CPU05|CPU06|CPU07|CPU08|CPU09|CPU10|CPU11|
 *
 * (5.1) ABT_SET_AFFINITY="{0,1,2,3},{4,5,6,7},{8,9,10,11}"
 *      This explicitly sets all the CPU IDs.
 * (5.2) ABT_SET_AFFINITY="{0,1,2,3}:3:4"
 *      This creates 3 CPU ID lists starting from {0,1,2,3}. The stride is "4",
 *      so the created lists are "{0,1,2,3}, {4,5,6,7}, {8,9,10,11}".
 *      Note that "{0,1,2,3}:3:1" (or "{0,1,2,3}:3") is wrong: they create
 *      "{0,1,2,3}, {2,3,4,5}, {3,4,5,6}" since the stride is 1.
 * (5.3) ABT_SET_AFFINITY="{0:4:1}:3:4"
 *      "{0:4:1}" means a CPU ID list of 4 CPU IDs that starts from 0 with a
 *      stride 1, so it is the same as "{0,1,2,3}".
 * (5.4) ABT_SET_AFFINITY="{0:4}:3:4"
 *      The default stride is 1.
 *
 * Example 6: bind ES with rank n to cores that have either of the following:
 *            CPU ID = (n * 2) % 6
 *            CPU ID = (n * 2) % 6 + 6
 *
 * |    ES0    |    ES1    |    ES2    |    ES3    |    ES4    |    ES5    |
 * |CPU00|CPU06|CPU01|CPU07|CPU02|CPU08|CPU03|CPU09|CPU04|CPU10|CPU05|CPU11|
 *
 * (6.1) ABT_SET_AFFINITY="{0,6},{1,7},{2,8},{3,9},{4,10},{5,11}"
 *      This explicitly sets all the CPU IDs.
 * (6.2) ABT_SET_AFFINITY="{0,6}:6:1"
 *      This creates 6 CPU ID lists starting from {0,6}. The stride is "1", so
 *      the created lists are "{0,6}, {1,7}, {2,8}, {3,9}, {4,10}, {5,11}".
 * (6.3) ABT_SET_AFFINITY="{0,6}:6"
 *      The default stride is 1.
 *
 * Example 7: bind ESs to cores except for those that have CPU ID = 0 or 1
 *
 * |           |                    ES0, ES1, ES2, ...                     |
 * |CPU00|CPU01|CPU02|CPU03|CPU04|CPU05|CPU06|CPU07|CPU08|CPU09|CPU10|CPU11|
 *
 * (7.1) ABT_SET_AFFINITY="{2,3,4,5,6,7,8,9,10,11}"
 *      This explicitly sets all the CPU IDs.
 * (7.2) ABT_SET_AFFINITY="{2:10:1}"
 *      "{2:10:1}" means a CPU ID list that has 10 CPU IDs starting from 2 with
 *      a stride 1, so it is the same as "{2,3,4,5,6,7,8,9,10,11}".
 * (7.3) ABT_SET_AFFINITY="{2:10}"
 *      The default stride is 1.
 */

#include "abti.h"
#include <unistd.h>

#ifdef HAVE_PTHREAD_SETAFFINITY_NP
#ifdef __FreeBSD__

#include <sys/param.h>
#include <sys/cpuset.h>
#include <pthread_np.h>
typedef cpuset_t cpu_set_t;

#else /* !__FreeBSD__ */

#define _GNU_SOURCE
#include <sched.h>

#endif
#endif /* HAVE_PTHREAD_SETAFFINITY_NP */

typedef struct {
    ABTD_affinity_cpuset initial_cpuset;
    size_t num_cpusets;
    ABTD_affinity_cpuset *cpusets;
} global_affinity;

static global_affinity g_affinity;

static inline int int_rem(int a, unsigned int b)
{
    /* Return x where a = n * b + x and 0 <= x < b */
    /* Because of ambiguity in the C specification, it uses a branch to check if
     * the result is positive. */
    int ret = (a % b) + b;
    return ret >= b ? (ret - b) : ret;
}

ABTU_ret_err static int get_num_cores(pthread_t native_thread, int *p_num_cores)
{
#ifdef HAVE_PTHREAD_SETAFFINITY_NP
    int i, num_cores = 0;
    /* Check the number of available cores by counting set bits. */
    cpu_set_t cpuset;
    int ret = pthread_getaffinity_np(native_thread, sizeof(cpu_set_t), &cpuset);
    if (ret)
        return ABT_ERR_OTHER;
    for (i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &cpuset)) {
            num_cores++;
        }
    }
    *p_num_cores = num_cores;
    return ABT_SUCCESS;
#else
    return ABT_ERR_FEATURE_NA;
#endif
}

ABTU_ret_err static int read_cpuset(pthread_t native_thread,
                                    ABTD_affinity_cpuset *p_cpuset)
{
#ifdef HAVE_PTHREAD_SETAFFINITY_NP
    cpu_set_t cpuset;
    int ret = pthread_getaffinity_np(native_thread, sizeof(cpu_set_t), &cpuset);
    if (ret)
        return ABT_ERR_OTHER;
    int i, j, num_cpuids = 0;
    for (i = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &cpuset))
            num_cpuids++;
    }
    p_cpuset->num_cpuids = num_cpuids;
    ret = ABTU_malloc(sizeof(int) * num_cpuids, (void **)&p_cpuset->cpuids);
    ABTI_CHECK_ERROR(ret);
    for (i = 0, j = 0; i < CPU_SETSIZE; i++) {
        if (CPU_ISSET(i, &cpuset))
            p_cpuset->cpuids[j++] = i;
    }
    return ABT_SUCCESS;
#else
    return ABT_ERR_FEATURE_NA;
#endif
}

ABTU_ret_err static int apply_cpuset(pthread_t native_thread,
                                     const ABTD_affinity_cpuset *p_cpuset)
{
#ifdef HAVE_PTHREAD_SETAFFINITY_NP
    size_t i;
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    for (i = 0; i < p_cpuset->num_cpuids; i++) {
        CPU_SET(int_rem(p_cpuset->cpuids[i], CPU_SETSIZE), &cpuset);
    }
    int ret = pthread_setaffinity_np(native_thread, sizeof(cpu_set_t), &cpuset);
    return ret == 0 ? ABT_SUCCESS : ABT_ERR_OTHER;
#else
    return ABT_ERR_FEATURE_NA;
#endif
}

void ABTD_affinity_init(const char *affinity_str)
{
    g_affinity.num_cpusets = 0;
    g_affinity.cpusets = NULL;
    g_affinity.initial_cpuset.cpuids = NULL;
    pthread_t self_native_thread = pthread_self();
    int i, ret;
    ret = get_num_cores(self_native_thread, &gp_ABTI_global->num_cores);
    if (ret != ABT_SUCCESS || gp_ABTI_global->num_cores == 0) {
        gp_ABTI_global->set_affinity = ABT_FALSE;
        return;
    }
    ret = read_cpuset(self_native_thread, &g_affinity.initial_cpuset);
    if (ret != ABT_SUCCESS) {
        gp_ABTI_global->set_affinity = ABT_FALSE;
        return;
    } else if (g_affinity.initial_cpuset.num_cpuids == 0) {
        ABTD_affinity_cpuset_destroy(&g_affinity.initial_cpuset);
        gp_ABTI_global->set_affinity = ABT_FALSE;
        return;
    }
    gp_ABTI_global->set_affinity = ABT_TRUE;
    ABTD_affinity_list *p_list = ABTD_affinity_list_create(affinity_str);
    if (p_list) {
        if (p_list->num == 0) {
            ABTD_affinity_list_free(p_list);
            p_list = NULL;
        }
    }
    if (p_list) {
        /* Create cpusets based on the affinity list.*/
        g_affinity.num_cpusets = p_list->num;
        ret = ABTU_calloc(g_affinity.num_cpusets, sizeof(ABTD_affinity_cpuset),
                          (void **)&g_affinity.cpusets);
        ABTI_ASSERT(ret == ABT_SUCCESS);
        for (i = 0; i < p_list->num; i++) {
            const ABTD_affinity_id_list *p_id_list = p_list->p_id_lists[i];
            int j, num_cpuids = 0, len_cpuids = 8;
            ret = ABTU_malloc(sizeof(int) * len_cpuids,
                              (void **)&g_affinity.cpusets[i].cpuids);
            ABTI_ASSERT(ret == ABT_SUCCESS);
            if (ABTI_IS_ERROR_CHECK_ENABLED && ret != ABT_SUCCESS) {
                ABTD_affinity_list_free(p_list);
                gp_ABTI_global->set_affinity = ABT_FALSE;
                return;
            }
            for (j = 0; j < p_id_list->num; j++) {
                int cpuid_i = int_rem(p_id_list->ids[j],
                                      g_affinity.initial_cpuset.num_cpuids);
                int cpuid = g_affinity.initial_cpuset.cpuids[cpuid_i];
                /* If it is unique, add it.*/
                int k, is_unique = 1;
                for (k = 0; k < num_cpuids; k++) {
                    if (g_affinity.cpusets[i].cpuids[k] == cpuid) {
                        is_unique = 0;
                        break;
                    }
                }
                if (is_unique) {
                    if (num_cpuids == len_cpuids) {
                        ret = ABTU_realloc(sizeof(int) * len_cpuids,
                                           sizeof(int) * len_cpuids * 2,
                                           (void **)&g_affinity.cpusets[i]
                                               .cpuids);
                        ABTI_ASSERT(ret == ABT_SUCCESS);
                        len_cpuids *= 2;
                    }
                    g_affinity.cpusets[i].cpuids[num_cpuids] = cpuid;
                    num_cpuids++;
                }
            }
            /* Adjust the size of cpuids. */
            if (num_cpuids != len_cpuids)
                ret = ABTU_realloc(sizeof(int) * len_cpuids,
                                   sizeof(int) * num_cpuids,
                                   (void **)&g_affinity.cpusets[i].cpuids);
            ABTI_ASSERT(ret == ABT_SUCCESS);
            g_affinity.cpusets[i].num_cpuids = num_cpuids;
        }
        ABTD_affinity_list_free(p_list);
    } else {
        /* Create default cpusets. */
        g_affinity.num_cpusets = g_affinity.initial_cpuset.num_cpuids;
        ret = ABTU_calloc(g_affinity.num_cpusets, sizeof(ABTD_affinity_cpuset),
                          (void **)&g_affinity.cpusets);
        ABTI_ASSERT(ret == ABT_SUCCESS);
        for (i = 0; i < g_affinity.num_cpusets; i++) {
            g_affinity.cpusets[i].num_cpuids = 1;
            ret = ABTU_malloc(sizeof(int) * g_affinity.cpusets[i].num_cpuids,
                              (void **)&g_affinity.cpusets[i].cpuids);
            ABTI_ASSERT(ret == ABT_SUCCESS);
            g_affinity.cpusets[i].cpuids[0] =
                g_affinity.initial_cpuset.cpuids[i];
        }
    }
}

void ABTD_affinity_finalize(void)
{
    pthread_t self_native_thread = pthread_self();
    if (gp_ABTI_global->set_affinity) {
        /* Set the affinity of the main native thread to the original one. */
        int abt_errno =
            apply_cpuset(self_native_thread, &g_affinity.initial_cpuset);
        /* Even if this cpuset apply fails, there is no way to handle it (e.g.,
         * possibly the CPU affinity policy has been changed while running
         * a user program.  Let's ignore this error. */
        (void)abt_errno;
    }
    /* Free g_afinity. */
    ABTD_affinity_cpuset_destroy(&g_affinity.initial_cpuset);
    int i;
    for (i = 0; i < g_affinity.num_cpusets; i++) {
        ABTD_affinity_cpuset_destroy(&g_affinity.cpusets[i]);
    }
    ABTU_free(g_affinity.cpusets);
    g_affinity.cpusets = NULL;
    g_affinity.num_cpusets = 0;
}

ABTU_ret_err int ABTD_affinity_cpuset_read(ABTD_xstream_context *p_ctx,
                                           ABTD_affinity_cpuset *p_cpuset)
{
    return read_cpuset(p_ctx->native_thread, p_cpuset);
}

ABTU_ret_err int
ABTD_affinity_cpuset_apply(ABTD_xstream_context *p_ctx,
                           const ABTD_affinity_cpuset *p_cpuset)
{
    return apply_cpuset(p_ctx->native_thread, p_cpuset);
}

ABTU_ret_err int ABTD_affinity_cpuset_apply_default(ABTD_xstream_context *p_ctx,
                                                    int rank)
{
    if (gp_ABTI_global->set_affinity) {
        ABTD_affinity_cpuset *p_cpuset =
            &g_affinity.cpusets[rank % g_affinity.num_cpusets];
        return apply_cpuset(p_ctx->native_thread, p_cpuset);
    }
    return ABT_SUCCESS;
}

void ABTD_affinity_cpuset_destroy(ABTD_affinity_cpuset *p_cpuset)
{
    if (p_cpuset) {
        ABTU_free(p_cpuset->cpuids);
        p_cpuset->cpuids = NULL;
    }
}
