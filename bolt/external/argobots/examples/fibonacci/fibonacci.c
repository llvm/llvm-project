/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * Notorious Fibonacci code that uses recursive parallelism.  Each scheduler has
 *  its own pool, and created ULTs are pushed to its local pool.  Pools are
 * shared among schedulers, so ULTs can be run on any execution streams by work
 * stealing.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <unistd.h>
#include <stdarg.h>
#include <abt.h>

#define DEFAULT_NUM_XSTREAMS 4
#define DEFAULT_N 10

ABT_pool *pools;

typedef struct {
    int n;
    int ret;
} fibonacci_arg_t;

void fibonacci(void *arg)
{
    int n = ((fibonacci_arg_t *)arg)->n;
    int *p_ret = &((fibonacci_arg_t *)arg)->ret;

    if (n <= 1) {
        *p_ret = 1;
    } else {
        fibonacci_arg_t child1_arg = { n - 1, 0 };
        fibonacci_arg_t child2_arg = { n - 2, 0 };
        int rank;
        ABT_xstream_self_rank(&rank);
        ABT_pool target_pool = pools[rank];
        ABT_thread child1;
        /* Calculate fib(n - 1). */
        ABT_thread_create(target_pool, fibonacci, &child1_arg,
                          ABT_THREAD_ATTR_NULL, &child1);
        /* Calculate fib(n - 2).  We do not create another ULT. */
        fibonacci(&child2_arg);
        ABT_thread_free(&child1);
        *p_ret = child1_arg.ret + child2_arg.ret;
    }
}

int fibonacci_seq(int n)
{
    if (n <= 1) {
        return 1;
    } else {
        int i;
        int fib_i1 = 1; /* Value of fib(i - 1) */
        int fib_i2 = 1; /* Value of fib(i - 2) */
        for (i = 3; i <= n; i++) {
            int tmp = fib_i1;
            fib_i1 = fib_i1 + fib_i2;
            fib_i2 = tmp;
        }
        return fib_i1 + fib_i2;
    }
}

int main(int argc, char **argv)
{
    int i, j;
    /* Read arguments. */
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    int n = DEFAULT_N;
    while (1) {
        int opt = getopt(argc, argv, "he:n:");
        if (opt == -1)
            break;
        switch (opt) {
            case 'e':
                num_xstreams = atoi(optarg);
                break;
            case 'n':
                n = atoi(optarg);
                break;
            case 'h':
            default:
                printf("Usage: ./fibonacci [-e NUM_XSTREAMS] [-n N]\n");
                return -1;
        }
    }

    /* Allocate memory. */
    ABT_xstream *xstreams =
        (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    ABT_sched *scheds = (ABT_sched *)malloc(sizeof(ABT_sched) * num_xstreams);

    /* Initialize Argobots. */
    ABT_init(argc, argv);

    /* Create pools. */
    for (i = 0; i < num_xstreams; i++) {
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE,
                              &pools[i]);
    }

    /* Create schedulers. */
    for (i = 0; i < num_xstreams; i++) {
        ABT_pool *tmp = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
        for (j = 0; j < num_xstreams; j++) {
            tmp[j] = pools[(i + j) % num_xstreams];
        }
        ABT_sched_create_basic(ABT_SCHED_DEFAULT, num_xstreams, tmp,
                               ABT_SCHED_CONFIG_NULL, &scheds[i]);
        free(tmp);
    }

    /* Set up a primary execution stream. */
    ABT_xstream_self(&xstreams[0]);
    ABT_xstream_set_main_sched(xstreams[0], scheds[0]);

    /* Create secondary execution streams. */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_create(scheds[i], &xstreams[i]);
    }

    fibonacci_arg_t arg = { n, 0 };
    fibonacci(&arg);
    int ret = arg.ret;
    int ans = fibonacci_seq(n);

    /* Join secondary execution streams. */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }

    /* Finalize Argobots. */
    ABT_finalize();

    /* Free allocated memory. */
    free(xstreams);
    free(pools);
    free(scheds);

    /* Check the results. */
    printf("Fibonacci(%d) = %d (ans: %d)\n", n, ret, ans);
    return ret != ans ? -1 : 0;
}
