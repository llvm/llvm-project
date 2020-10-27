/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * Base implementation: stencil_naive.c
 *
 * Single-threaded 2D stencil code.  The does the same computation as does
 * stencil_naive.  However, the programmer can easily parallelize this version
 * (compared with stencil_naive.c) since the kernel is computed per block.
 * Particularly, one needs to parallelize the execution of kernel().
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
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
    int t;
    /* Read arguments. */
    int read_arg_ret =
        read_args(argc, argv, &num_blocksX, &num_blocksY, &blocksize,
                  &num_iters, &num_xstreams, &validate);
    if (read_arg_ret != 0) {
        return -1;
    }

    /* Allocate memory. */
    double *values_old = (double *)malloc(sizeof(double) * WIDTH * HEIGHT);
    double *values_new = (double *)malloc(sizeof(double) * WIDTH * HEIGHT);
    kernel_arg_t *kernel_args = (kernel_arg_t *)malloc(
        sizeof(kernel_arg_t) * num_blocksX * num_blocksY);

    /* Initialize grid values. */
    init_values(values_old, values_new, num_blocksX, num_blocksY, blocksize);

    /* Main iteration loop. */
    for (t = 0; t < num_iters; t++) {
        int blockX, blockY;
        for (blockX = 0; blockX < num_blocksX; blockX++) {
            for (blockY = 0; blockY < num_blocksY; blockY++) {
                kernel_arg_t *p_kernel_arg =
                    &kernel_args[blockX + blockY * num_blocksX];
                p_kernel_arg->values_old = values_old;
                p_kernel_arg->values_new = values_new;
                p_kernel_arg->blockX = blockX;
                p_kernel_arg->blockY = blockY;
                kernel(p_kernel_arg);
            }
        }
        /* Swap values_old and values_new. */
        double *values_tmp = values_new;
        values_new = values_old;
        values_old = values_tmp;
    }

    /* Validate results.  values_old has the latest values. */
    int validate_ret = 0;
    if (validate) {
        validate_ret = validate_values(values_old, num_blocksX, num_blocksY,
                                       blocksize, num_iters);
    }

    /* Free allocated memory. */
    free(values_old);
    free(values_new);
    free(kernel_args);

    if (validate_ret != 0) {
        printf("Validation failed.\n");
        return -1;
    } else if (validate) {
        printf("Validation succeeded.\n");
    }
    return 0;
}
