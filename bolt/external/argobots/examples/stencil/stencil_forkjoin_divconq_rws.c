/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * Base implementation: stencil_forkjoin_divconq.c
 *
 * Parallel 2D stencil code based on a fork-join strategy.  In every iteration,
 * it creates as many ULTs as the number of blocks (num_blocksX * num_blocksY)
 * and frees them.  Fork-join in each iteration is needed for halo
 * synchronization.
 *
 * This divide-and-conquer version creates ULTs in a divide-and-conquer manner.
 * Each ULT is in charge of [blockX_from, blockX_to) x [blockY_from, blockY_to)
 * blocks.  If either X or Y side is longer than 1, it divides that side by
 * two and creates corresponding child ULTs; since it is applied to both X and Y
 * axes, each ULT has at most four children.  If the lengths of both sides
 * are 1, the ULT becomes a leaf node and runs the five-point stencil kernel.
 *
 * In this version, ULTs are scheduled in a typical random-work-stealing manner.
 * Created ULTs are pushed to the local pool.  A scheduler first tries to get a
 * ULT from its local pool unless it is empty; otherwise a scheduler tries to
 * steal a ULT from another pool that is randomly chosen.
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <abt.h>
#include "stencil_helper.h"

/* Global variables. */
int num_blocksX, num_blocksY;
int blocksize;
int num_iters;
int num_xstreams;
int validate;

typedef struct {
    double *values_old;
    double *values_new;
    int blockX_from;
    int blockY_from;
    int blockX_to;
    int blockY_to;
    ABT_pool *pools;
} thread_arg_t;

void thread(void *arg)
{
    double *values_old = ((thread_arg_t *)arg)->values_old;
    double *values_new = ((thread_arg_t *)arg)->values_new;
    int blockX_from = ((thread_arg_t *)arg)->blockX_from;
    int blockY_from = ((thread_arg_t *)arg)->blockY_from;
    int blockX_to = ((thread_arg_t *)arg)->blockX_to;
    int blockY_to = ((thread_arg_t *)arg)->blockY_to;
    ABT_pool *pools = ((thread_arg_t *)arg)->pools;

    if (blockX_to - blockX_from == 1 && blockY_to - blockY_from == 1) {
        /* Run stencil kernel. */
        int x, y;
        for (y = blockY_from * blocksize; y < blockY_to * blocksize; y++) {
            for (x = blockX_from * blocksize; x < blockX_to * blocksize; x++) {
                values_new[INDEX(x, y)] =
                    values_old[INDEX(x, y)] * (1.0 / 2.0) +
                    (values_old[INDEX(x + 1, y)] + values_old[INDEX(x - 1, y)] +
                     values_old[INDEX(x, y + 1)] +
                     values_old[INDEX(x, y - 1)]) *
                        (1.0 / 8.0);
            }
        }
    } else {
        /* Divide the region and create child threads (maximum four). */
        ABT_thread threads[4];
        thread_arg_t thread_args[4];
        int xdiv, ydiv;
        for (ydiv = 0; ydiv < 2; ydiv++) {
            for (xdiv = 0; xdiv < 2; xdiv++) {
                int index = xdiv + ydiv * 2;
                thread_args[index].values_old = values_old;
                thread_args[index].values_new = values_new;
                if (xdiv == 0) {
                    thread_args[index].blockX_from = blockX_from;
                    thread_args[index].blockX_to =
                        blockX_from + (blockX_to - blockX_from) / 2;
                } else {
                    thread_args[index].blockX_from =
                        blockX_from + (blockX_to - blockX_from) / 2;
                    thread_args[index].blockX_to = blockX_to;
                }
                if (ydiv == 0) {
                    thread_args[index].blockY_from = blockY_from;
                    thread_args[index].blockY_to =
                        blockY_from + (blockY_to - blockY_from) / 2;
                } else {
                    thread_args[index].blockY_from =
                        blockY_from + (blockY_to - blockY_from) / 2;
                    thread_args[index].blockY_to = blockY_to;
                }
                thread_args[index].pools = pools;
                if (thread_args[index].blockX_to -
                            thread_args[index].blockX_from !=
                        0 &&
                    thread_args[index].blockY_to -
                            thread_args[index].blockY_from !=
                        0) {
                    /* Push a ULT to the local pool (pools[rank]). */
                    int rank;
                    ABT_xstream_self_rank(&rank);
                    ABT_thread_create(pools[rank], thread, &thread_args[index],
                                      ABT_THREAD_ATTR_NULL, &threads[index]);
                }
            }
        }
        /* Join child threads. */
        for (ydiv = 0; ydiv < 2; ydiv++) {
            for (xdiv = 0; xdiv < 2; xdiv++) {
                int index = xdiv + ydiv * 2;
                if (thread_args[index].blockX_to -
                            thread_args[index].blockX_from !=
                        0 &&
                    thread_args[index].blockY_to -
                            thread_args[index].blockY_from !=
                        0) {
                    ABT_thread_free(&threads[index]);
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    int i, j, t;
    /* Read arguments. */
    int read_arg_ret =
        read_args(argc, argv, &num_blocksX, &num_blocksY, &blocksize,
                  &num_iters, &num_xstreams, &validate);
    if (read_arg_ret != 0) {
        return -1;
    }

    /* Allocate memory. */
    ABT_xstream *xstreams =
        (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    ABT_pool *pools = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
    ABT_sched *scheds = (ABT_sched *)malloc(sizeof(ABT_sched) * num_xstreams);
    double *values_old = (double *)malloc(sizeof(double) * WIDTH * HEIGHT);
    double *values_new = (double *)malloc(sizeof(double) * WIDTH * HEIGHT);

    /* Initialize grid values. */
    init_values(values_old, values_new, num_blocksX, num_blocksY, blocksize);

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
        ABT_sched_create_basic(ABT_SCHED_RANDWS, num_xstreams, tmp,
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

    /* Iterates stencil computation. */
    for (t = 0; t < num_iters; t++) {
        thread_arg_t thread_arg;
        thread_arg.values_old = values_old;
        thread_arg.values_new = values_new;
        thread_arg.blockX_from = 0;
        thread_arg.blockY_from = 0;
        thread_arg.blockX_to = num_blocksX;
        thread_arg.blockY_to = num_blocksY;
        thread_arg.pools = pools;
        thread(&thread_arg);
        /* Swap values_old and values_new. */
        double *values_tmp = values_new;
        values_new = values_old;
        values_old = values_tmp;
    }

    /* Join secondary execution streams. */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }

    /* Finalize Argobots */
    ABT_finalize();

    /* Validate results.  values_old has the latest values. */
    int validate_ret = 0;
    if (validate) {
        validate_ret = validate_values(values_old, num_blocksX, num_blocksY,
                                       blocksize, num_iters);
    }

    /* Free allocated memory. */
    free(xstreams);
    free(pools);
    free(scheds);
    free(values_old);
    free(values_new);

    if (validate_ret != 0) {
        printf("Validation failed.\n");
        return -1;
    } else if (validate) {
        printf("Validation succeeded.\n");
    }
    return 0;
}
