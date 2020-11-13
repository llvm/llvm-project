/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * Very basic sequential five-point 2D stencil code.  The grid contains
 * num_blocksX * num_blocksY blocks, each of which has blocksize x blocksize
 * points, so in total, the grid contains (num_blocksX * blocksize) x
 * (num_blocksY * blocksize) points.
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

    /* Initialize grid values. */
    init_values(values_old, values_new, num_blocksX, num_blocksY, blocksize);

    /* Main iteration loop. */
    for (t = 0; t < num_iters; t++) {
        int x, y;
        for (y = 0; y < num_blocksY * blocksize; y++) {
            for (x = 0; x < num_blocksX * blocksize; x++) {
                values_new[INDEX(x, y)] =
                    values_old[INDEX(x, y)] * (1.0 / 2.0) +
                    (values_old[INDEX(x + 1, y)] + values_old[INDEX(x - 1, y)] +
                     values_old[INDEX(x, y + 1)] +
                     values_old[INDEX(x, y - 1)]) *
                        (1.0 / 8.0);
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

    if (validate_ret != 0) {
        printf("Validation failed.\n");
        return -1;
    } else if (validate) {
        printf("Validation succeeded.\n");
    }
    return 0;
}
