/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_EVENTUAL_H_INCLUDED
#define ABTI_EVENTUAL_H_INCLUDED

/* Inlined functions for Eventual */

static inline ABTI_eventual *ABTI_eventual_get_ptr(ABT_eventual eventual)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_eventual *p_eventual;
    if (eventual == ABT_EVENTUAL_NULL) {
        p_eventual = NULL;
    } else {
        p_eventual = (ABTI_eventual *)eventual;
    }
    return p_eventual;
#else
    return (ABTI_eventual *)eventual;
#endif
}

static inline ABT_eventual ABTI_eventual_get_handle(ABTI_eventual *p_eventual)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_eventual h_eventual;
    if (p_eventual == NULL) {
        h_eventual = ABT_EVENTUAL_NULL;
    } else {
        h_eventual = (ABT_eventual)p_eventual;
    }
    return h_eventual;
#else
    return (ABT_eventual)p_eventual;
#endif
}

#endif /* ABTI_EVENTUAL_H_INCLUDED */
