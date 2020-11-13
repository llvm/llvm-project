/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include "abt.h"
#include "abttest.h"

#define DEFAULT_NUM_XSTREAMS 4

int main(int argc, char *argv[])
{
    ABT_xstream *xstreams;
    int num_xstreams = DEFAULT_NUM_XSTREAMS;
    int i, ret;
    int rank;

    /* Initialize */
    ATS_read_args(argc, argv);
    if (argc > 1) {
        num_xstreams = ATS_get_arg_val(ATS_ARG_N_ES);
    }
    ATS_init(argc, argv, num_xstreams);

    ATS_printf(1, "# of ESs: %d\n", num_xstreams);

    xstreams = (ABT_xstream *)malloc(sizeof(ABT_xstream) * num_xstreams);
    assert(xstreams != NULL);

    /* Create Execution Streams */
    ret = ABT_xstream_self(&xstreams[0]);
    ATS_ERROR(ret, "ABT_xstream_self");
    for (i = 1; i < num_xstreams; i++) {
        rank = num_xstreams - i;
        ret = ABT_xstream_create_with_rank(ABT_SCHED_NULL, rank, &xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_create_with_rank");
    }

    /* Check the rank of each ES */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_get_rank(xstreams[i], &rank);
        ATS_ERROR(ret, "ABT_xstream_get_rank");
        assert(rank == (num_xstreams - i));
    }

    /* Test an invalid rank, which is already taken */
    ABT_bool is_check_error = ABT_FALSE;
    ret = ABT_info_query_config(ABT_INFO_QUERY_KIND_ENABLED_CHECK_ERROR,
                                (void *)&is_check_error);
    ATS_ERROR(ret, "ABT_info_query_config");
    if (is_check_error) {
        ABT_xstream tmp;
        ret = ABT_xstream_create_with_rank(ABT_SCHED_NULL, 0, &tmp);
        assert(ret == ABT_ERR_INV_XSTREAM_RANK);
    }

    /* Join and free ESs */
    for (i = 1; i < num_xstreams; i++) {
        ret = ABT_xstream_join(xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_join");
        ret = ABT_xstream_free(&xstreams[i]);
        ATS_ERROR(ret, "ABT_xstream_free");
    }

    /* Finalize */
    ret = ATS_finalize(0);

    free(xstreams);

    return ret;
}
