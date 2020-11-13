/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * Creates multiple execution streams and runs ULTs on these execution streams.
 * Users can change the number of execution streams and the number of ULT via
 * arguments. Each ULT prints its ID.
 */

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdarg.h>
#include <abt.h>

#define DEFAULT_NUM_XSTREAMS 2
#define DEFAULT_NUM_THREADS 8

typedef struct {
    int tid;
} thread_arg_t;

void hello_world(void *arg)
{
    int tid = ((thread_arg_t *)arg)->tid;
    printf("Hello world! (thread = %d)\n", tid);
}

int main(int argc, char **argv)
{
    int i;
    /* Read arguments. */
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    int num_threads = DEFAULT_NUM_THREADS;
    while (1) {
        int opt = getopt(argc, argv, "he:n:");
        if (opt == -1)
            break;
        switch (opt) {
            case 'e':
                num_xstreams = atoi(optarg);
                break;
            case 'n':
                num_threads = atoi(optarg);
                break;
            case 'h':
            default:
                printf("Usage: ./hello_world [-e NUM_XSTREAMS] "
                       "[-n NUM_THREADS]\n");
                return -1;
        }
    }

    /* Allocate memory. */
    ABT_xstream *xstreams =
        (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    ABT_pool *pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    ABT_thread *threads =
        (ABT_thread *)malloc(sizeof(ABT_thread) * num_threads);
    thread_arg_t *thread_args =
        (thread_arg_t *)malloc(sizeof(thread_arg_t) * num_threads);

    /* Initialize Argobots. */
    ABT_init(argc, argv);

    /* Get a primary execution stream. */
    ABT_xstream_self(&xstreams[0]);

    /* Create secondary execution streams. */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_create(ABT_SCHED_NULL, &xstreams[i]);
    }

    /* Get default pools. */
    for (i = 0; i < num_xstreams; i++) {
        ABT_xstream_get_main_pools(xstreams[i], 1, &pools[i]);
    }

    /* Create ULTs. */
    for (i = 0; i < num_threads; i++) {
        int pool_id = i % num_xstreams;
        thread_args[i].tid = i;
        ABT_thread_create(pools[pool_id], hello_world, &thread_args[i],
                          ABT_THREAD_ATTR_NULL, &threads[i]);
    }

    /* Join and free ULTs. */
    for (i = 0; i < num_threads; i++) {
        ABT_thread_free(&threads[i]);
    }

    /* Join and free secondary execution streams. */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }

    /* Finalize Argobots. */
    ABT_finalize();

    /* Free allocated memory. */
    free(xstreams);
    free(pools);
    free(threads);
    free(thread_args);

    return 0;
}
