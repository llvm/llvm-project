/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * Base implementation: stencil_forkjoin.c
 *
 * Parallel 2D stencil code based on a fork-join strategy.  In every iteration,
 * it creates as many ULTs as the number of blocks (num_blocksX * num_blocksY)
 * and frees them.  Fork-join in each iteration is needed for halo
 * synchronization.
 *
 * Unlike stencil_forkjoin.c, pools are shared among schedulers, so ULTs are
 * dynamically scheduled by work stealing.
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
    int blockX;
    int blockY;
} kernel_arg_t;

void kernel(void *arg)
{
    int x, y;
    double *values_old = ((kernel_arg_t *)arg)->values_old;
    double *values_new = ((kernel_arg_t *)arg)->values_new;
    int blockX = ((kernel_arg_t *)arg)->blockX;
    int blockY = ((kernel_arg_t *)arg)->blockY;
    for (y = blockY * blocksize; y < (blockY + 1) * blocksize; y++) {
        for (x = blockX * blocksize; x < (blockX + 1) * blocksize; x++) {
            values_new[INDEX(x, y)] =
                values_old[INDEX(x, y)] * (1.0 / 2.0) +
                (values_old[INDEX(x + 1, y)] + values_old[INDEX(x - 1, y)] +
                 values_old[INDEX(x, y + 1)] + values_old[INDEX(x, y - 1)]) *
                    (1.0 / 8.0);
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
    ABT_thread *threads =
        (ABT_thread *)malloc(sizeof(ABT_thread) * num_blocksX * num_blocksY);
    kernel_arg_t *kernel_args = (kernel_arg_t *)malloc(
        sizeof(kernel_arg_t) * num_blocksX * num_blocksY);

    /* Initialize grids values. */
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

    /* Iterates stencil computation. */
    for (t = 0; t < num_iters; t++) {
        /* Create ULTs. */
        int blockX, blockY;
        for (blockX = 0; blockX < num_blocksX; blockX++) {
            for (blockY = 0; blockY < num_blocksY; blockY++) {
                int index = blockX + blockY * num_blocksX;
                kernel_arg_t *p_kernel_arg = &kernel_args[index];
                p_kernel_arg->values_old = values_old;
                p_kernel_arg->values_new = values_new;
                p_kernel_arg->blockX = blockX;
                p_kernel_arg->blockY = blockY;
                int pool_id = index % num_xstreams;
                ABT_thread_create(pools[pool_id], kernel, p_kernel_arg,
                                  ABT_THREAD_ATTR_NULL, &threads[index]);
            }
        }
        /* Join and free ULTs. */
        for (blockX = 0; blockX < num_blocksX; blockX++) {
            for (blockY = 0; blockY < num_blocksY; blockY++) {
                int index = blockX + blockY * num_blocksX;
                ABT_thread_free(&threads[index]);
            }
        }

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

    /* Finalize Argobots. */
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
    free(threads);
    free(kernel_args);

    if (validate_ret != 0) {
        printf("Validation failed.\n");
        return -1;
    } else if (validate) {
        printf("Validation succeeded.\n");
    }
    return 0;
}
