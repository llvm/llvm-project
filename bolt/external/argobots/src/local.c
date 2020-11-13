/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

static ABTI_local *local_get_local_internal(void)
{
    return lp_ABTI_local;
}

static void local_set_xstream_internal(ABTI_xstream *p_local_xstream)
{
    lp_ABTI_local = (ABTI_local *)p_local_xstream;
}

static void *local_get_local_ptr_internal(void)
{
    return (void *)&lp_ABTI_local;
}

ABTI_local_func gp_ABTI_local_func = { { 0 },
                                       local_get_local_internal,
                                       local_set_xstream_internal,
                                       local_get_local_ptr_internal,
                                       { 0 } };
/* ES Local Data */
ABTD_XSTREAM_LOCAL ABTI_local *lp_ABTI_local = NULL;
