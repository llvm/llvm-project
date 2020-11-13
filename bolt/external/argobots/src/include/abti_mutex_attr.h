/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_MUTEX_ATTR_H_INCLUDED
#define ABTI_MUTEX_ATTR_H_INCLUDED

/* Inlined functions for mutex attributes */

static inline ABTI_mutex_attr *ABTI_mutex_attr_get_ptr(ABT_mutex_attr attr)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_mutex_attr *p_attr;
    if (attr == ABT_MUTEX_ATTR_NULL) {
        p_attr = NULL;
    } else {
        p_attr = (ABTI_mutex_attr *)attr;
    }
    return p_attr;
#else
    return (ABTI_mutex_attr *)attr;
#endif
}

static inline ABT_mutex_attr ABTI_mutex_attr_get_handle(ABTI_mutex_attr *p_attr)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_mutex_attr h_attr;
    if (p_attr == NULL) {
        h_attr = ABT_MUTEX_ATTR_NULL;
    } else {
        h_attr = (ABT_mutex_attr)p_attr;
    }
    return h_attr;
#else
    return (ABT_mutex_attr)p_attr;
#endif
}

#endif /* ABTI_MUTEX_ATTR_H_INCLUDED */
