/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

/*
 * Base implementation: stencil_forkjoin_divconq_rws.c
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
 *         pools[0]    ..   pools[n/2]     pools[n/2+1]    ..   pools[n-1]
 *        xstreams[0]  ..  xstreams[n/2]  xstreams[n/2+1]  ..  xstreams[n-1]
 * L2    |------------------------------------------------------------------|
 * L1    |------------------------------||----------------------------------|
 * Local |-----------||..||-------------||---------------||..||-------------|
 *                                                            n = num_xstreams
 * In this version, ULTs are scheduled based on a simple hierarchical random-
 * work-stealing method.  Created ULTs are pushed to the local pool and
 * a scheduler tries to get a ULT from its local pool unless it is empty.  If
 * the local pool is empty, the scheduler first tries to get a ULT from one of
 * level-1 pools that are belonging to execution streams closer to the current
 * execution stream.  If work stealing fails several times, the scheduler tries
 * to steal a ULT from one of all pools (i.e., including level-2 pools).
 *
 * To really improve performance, users need to pin execution streams to cores
 * by setting affinity (i.e., configure Argobots with --enable-affinity).
 */

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>
#include <time.h>
#include <abt.h>
#include "stencil_helper.h"

/* Global variables */
int num_blocksX, num_blocksY;
int blocksize;
int num_iters;
int num_xstreams;
int validate;

int sched_init(ABT_sched sched, ABT_sched_config config)
{
    return ABT_SUCCESS;
}

void sched_run(ABT_sched sched)
{
    int work_count = 0, num_pools;
    ABT_pool *pools;
    unsigned seed = time(NULL);

    ABT_sched_get_num_pools(sched, &num_pools);
    pools = (ABT_pool *)malloc(num_pools * sizeof(ABT_pool));
    ABT_sched_get_pools(sched, num_pools, 0, pools);

    while (1) {
        ABT_unit unit;
        /* Try to pop a ULT from a local pool*/
        ABT_pool_pop(pools[0], &unit);
        if (unit != ABT_UNIT_NULL) {
            ABT_xstream_run_unit(unit, pools[0]);
            goto EVENT_CHECK;
        }
        if (num_pools > 1) {
            /* If failed, try to pop a ULT from level-1 pools several times */
            int repeat = 0;
            while (repeat++ < 2 && unit == ABT_UNIT_NULL) {
                unsigned rand_val = rand_r(&seed);
                int victim = rand_val % (num_pools / 2);
                ABT_pool_pop(pools[victim], &unit);
            }
            if (unit != ABT_UNIT_NULL) {
                ABT_unit_set_associated_pool(unit, pools[0]);
                ABT_xstream_run_unit(unit, pools[0]);
                goto EVENT_CHECK;
            }
            /* If failed, try to pop a ULT from all the pools */
            {
                unsigned rand_val = rand_r(&seed);
                int victim = rand_val % num_pools;
                ABT_pool_pop(pools[victim], &unit);
            }
            if (unit != ABT_UNIT_NULL) {
                ABT_unit_set_associated_pool(unit, pools[0]);
                ABT_xstream_run_unit(unit, pools[0]);
            }
        }
    EVENT_CHECK:
        if (++work_count % 4096 == 0) {
            ABT_bool stop;
            ABT_xstream_check_events(sched);
            ABT_sched_has_to_stop(sched, &stop);
            if (stop == ABT_TRUE) {
                break;
            }
        }
    }
    free(pools);
}

int sched_free(ABT_sched sched)
{
    return ABT_SUCCESS;
}

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
        /* Run the stencil kernel. */
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
    /* Initialize grids. */
    init_values(values_old, values_new, num_blocksX, num_blocksY, blocksize);

    /* Initialize Argobots. */
    ABT_init(argc, argv);

    /* Create pools. */
    for (i = 0; i < num_xstreams; i++) {
        ABT_pool_create_basic(ABT_POOL_FIFO, ABT_POOL_ACCESS_MPMC, ABT_TRUE,
                              &pools[i]);
    }

    /* Create schedulers. */
    ABT_sched_def sched_def = {
        .type = ABT_SCHED_TYPE_ULT,
        .init = sched_init,
        .run = sched_run,
        .free = sched_free,
        .get_migr_pool = NULL,
    };
    for (i = 0; i < num_xstreams; i++) {
        ABT_pool *tmp = (ABT_pool *)malloc(sizeof(ABT_pool) * num_xstreams);
        int pool_index = 0;
        tmp[pool_index++] = pools[i];
        if (i < num_xstreams / 2) {
            /* The first num_xstreams / 2 pools are considered level-1 pools. */
            for (j = 0; j < num_xstreams; j++) {
                if (i != j) {
                    tmp[pool_index++] = pools[j];
                }
            }
        } else {
            /* The other pools are considered level-2 pools. */
            for (j = num_xstreams / 2; j < num_xstreams; j++) {
                if (i != j) {
                    tmp[pool_index++] = pools[j];
                }
            }
            for (j = 0; j < num_xstreams / 2; j++) {
                tmp[pool_index++] = pools[j];
            }
        }
        ABT_sched_create(&sched_def, num_xstreams, tmp, ABT_SCHED_CONFIG_NULL,
                         &scheds[i]);
        free(tmp);
    }

    /* Set up a primary execution stream. */
    ABT_xstream_self(&xstreams[0]);
    ABT_xstream_set_main_sched(xstreams[0], scheds[0]);

    /* Create secondary execution streams. */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_create(scheds[i], &xstreams[i]);
    }

    /* Main loop */
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
        /* Swap values_old and values_new */
        double *values_tmp = values_new;
        values_new = values_old;
        values_old = values_tmp;
    }

    /* Join secondary execution streams. */
    for (i = 1; i < num_xstreams; i++) {
        ABT_xstream_join(xstreams[i]);
        ABT_xstream_free(&xstreams[i]);
    }

    /* Free secondary schedulers: the scheduler of the primary execution stream
     * may not be freed since it is now scheduling this main ULT (a thread that
     * calls ABT_initialize() at this point. */
    for (i = 1; i < num_xstreams; i++) {
        ABT_sched_free(&scheds[i]);
    }

    /* Finalize Argobots */
    ABT_finalize();

    /* Validate results. */
    int validate_ret = 0;
    if (validate) {
        validate_ret = validate_values(values_old, num_blocksX, num_blocksY,
                                       blocksize, num_iters);
    }

    /* Free allocated memory */
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
