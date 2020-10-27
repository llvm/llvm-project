/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */
/*
 *
 * Base implementation: stencil_barrier.c
 * See also: stencil_depend_yield.c
 *
 * Parallel 2D stencil code that uses future-based synchronization.  First, it
 * spawns as many ULTs as the number of blocks (num_blocksX * num_blocksY).  For
 * halo synchronization, a ULT waits for completion of all the four neighbors'
 * computations on future objects; the ULT will be resumed if all the neighbors
 * completed the computation of an iteration t - 1 and set the corresponding
 * future.
 *
 * This exposes more parallelism than the barrier method; in this method, each
 * ULT only waits for completion of previous computations of four neighbor
 * blocks whereas, with a barrier synchronization, each ULT needs to wait for
 * completion of previous computations of all blocks.
 *
 * ULTs cannot be simply replaced with tasklets in this case since future
 * operations in kernel() are synchronization operations that cannot be called
 * on a tasklet.
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
    ABT_future self_progress_future_old, self_progress_future_new;
    ABT_future neighbor_progress_futures_old[4],
        neighbor_progress_futures_new[4];
} kernel_arg_t;

void kernel(void *arg)
{
    int i, t;
    double *values_old = ((kernel_arg_t *)arg)->values_old;
    double *values_new = ((kernel_arg_t *)arg)->values_new;
    int blockX = ((kernel_arg_t *)arg)->blockX;
    int blockY = ((kernel_arg_t *)arg)->blockY;
    ABT_future self_progress_future_old =
        ((kernel_arg_t *)arg)->self_progress_future_old;
    ABT_future self_progress_future_new =
        ((kernel_arg_t *)arg)->self_progress_future_new;
    ABT_future *neighbor_progress_futures_old =
        ((kernel_arg_t *)arg)->neighbor_progress_futures_old;
    ABT_future *neighbor_progress_futures_new =
        ((kernel_arg_t *)arg)->neighbor_progress_futures_new;
    /* Iterates stencil computation. */
    for (t = 0; t < num_iters; t++) {
        /* Check progress of neighbors. */
        if (t != 0) {
            ABT_future_wait(self_progress_future_old);
            ABT_future_reset(self_progress_future_old);
        }
        /* Run the stencil kernel. */
        int x, y;
        for (y = blockY * blocksize; y < (blockY + 1) * blocksize; y++) {
            for (x = blockX * blocksize; x < (blockX + 1) * blocksize; x++) {
                values_new[INDEX(x, y)] =
                    values_old[INDEX(x, y)] * (1.0 / 2.0) +
                    (values_old[INDEX(x + 1, y)] + values_old[INDEX(x - 1, y)] +
                     values_old[INDEX(x, y + 1)] +
                     values_old[INDEX(x, y - 1)]) *
                        (1.0 / 8.0);
            }
        }
        /* Set futures of neighbors. */
        for (i = 0; i < 4; i++) {
            if (neighbor_progress_futures_new[i] != ABT_FUTURE_NULL) {
                ABT_future_set(neighbor_progress_futures_new[i], NULL);
            }
        }

        /* Swap values_old and values_new. */
        double *values_tmp = values_new;
        values_new = values_old;
        values_old = values_tmp;
        ABT_future self_progress_future_tmp = self_progress_future_new;
        self_progress_future_new = self_progress_future_old;
        self_progress_future_old = self_progress_future_tmp;
        ABT_future *neighbor_progress_futures_tmp =
            neighbor_progress_futures_new;
        neighbor_progress_futures_new = neighbor_progress_futures_old;
        neighbor_progress_futures_old = neighbor_progress_futures_tmp;
    }
}

int main(int argc, char **argv)
{
    int i, blockX, blockY;
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
    double *values_old = (double *)malloc(sizeof(double) * WIDTH * HEIGHT);
    double *values_new = (double *)malloc(sizeof(double) * WIDTH * HEIGHT);
    ABT_thread *threads =
        (ABT_thread *)malloc(sizeof(ABT_thread) * num_blocksX * num_blocksY);
    ABT_future *progress_futures_old =
        (ABT_future *)malloc(sizeof(ABT_future) * num_blocksX * num_blocksY);
    ABT_future *progress_futures_new =
        (ABT_future *)malloc(sizeof(ABT_future) * num_blocksX * num_blocksY);
    kernel_arg_t *kernel_args = (kernel_arg_t *)malloc(
        sizeof(kernel_arg_t) * num_blocksX * num_blocksY);

    /* Initialize grid values. */
    init_values(values_old, values_new, num_blocksX, num_blocksY, blocksize);

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

    /* Create futures. */
    for (blockX = 0; blockX < num_blocksX; blockX++) {
        for (blockY = 0; blockY < num_blocksY; blockY++) {
            int index = blockX + blockY * num_blocksX;
            int num_neighbors = 0;
            const int dirX[] = { -1, 1, 0, 0 };
            const int dirY[] = { 0, 0, -1, 1 };
            for (i = 0; i < 4; i++) {
                int neighborX = blockX + dirX[i];
                int neighborY = blockY + dirY[i];
                if (0 <= neighborX && neighborX < num_blocksX &&
                    0 <= neighborY && neighborY < num_blocksY) {
                    num_neighbors++;
                }
            }
            ABT_future_create(num_neighbors, 0, &progress_futures_old[index]);
            ABT_future_create(num_neighbors, 0, &progress_futures_new[index]);
        }
    }

    /* Create ULTs. */
    for (blockX = 0; blockX < num_blocksX; blockX++) {
        for (blockY = 0; blockY < num_blocksY; blockY++) {
            int index = blockX + blockY * num_blocksX;
            kernel_arg_t *p_kernel_arg = &kernel_args[index];
            p_kernel_arg->values_old = values_old;
            p_kernel_arg->values_new = values_new;
            p_kernel_arg->blockX = blockX;
            p_kernel_arg->blockY = blockY;
            p_kernel_arg->self_progress_future_old =
                progress_futures_old[index];
            p_kernel_arg->self_progress_future_new =
                progress_futures_new[index];
            const int dirX[] = { -1, 1, 0, 0 };
            const int dirY[] = { 0, 0, -1, 1 };
            for (i = 0; i < 4; i++) {
                int neighborX = blockX + dirX[i];
                int neighborY = blockY + dirY[i];
                if (0 <= neighborX && neighborX < num_blocksX &&
                    0 <= neighborY && neighborY < num_blocksY) {
                    int neighbor_index = neighborX + neighborY * num_blocksX;
                    p_kernel_arg->neighbor_progress_futures_old[i] =
                        progress_futures_old[neighbor_index];
                    p_kernel_arg->neighbor_progress_futures_new[i] =
                        progress_futures_new[neighbor_index];
                } else {
                    /* There is no grid. */
                    p_kernel_arg->neighbor_progress_futures_old[i] =
                        ABT_FUTURE_NULL;
                    p_kernel_arg->neighbor_progress_futures_new[i] =
                        ABT_FUTURE_NULL;
                }
            }
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

    /* Free futures. */
    for (i = 0; i < num_blocksX * num_blocksY; i++) {
        ABT_future_free(&progress_futures_old[i]);
        ABT_future_free(&progress_futures_new[i]);
    }

    /* Swap values_old and values_new for validation. */
    if (num_iters % 2 == 1) {
        /* num_iters = 2N    : the latest is written to values_old. */
        /*           = 2N + 1: the latest is written to values_new. */
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
    free(values_old);
    free(values_new);
    free(threads);
    free(progress_futures_old);
    free(progress_futures_new);
    free(kernel_args);

    if (validate_ret != 0) {
        printf("Validation failed.\n");
        return -1;
    } else if (validate) {
        printf("Validation succeeded.\n");
    }
    return 0;
}
