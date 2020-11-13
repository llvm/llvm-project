/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_THREAD_ATTR_H_INCLUDED
#define ABTI_THREAD_ATTR_H_INCLUDED

/* Inlined functions for ULT Attributes */

static inline ABTI_thread_attr *ABTI_thread_attr_get_ptr(ABT_thread_attr attr)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_thread_attr *p_attr;
    if (attr == ABT_THREAD_ATTR_NULL) {
        p_attr = NULL;
    } else {
        p_attr = (ABTI_thread_attr *)attr;
    }
    return p_attr;
#else
    return (ABTI_thread_attr *)attr;
#endif
}

static inline ABT_thread_attr
ABTI_thread_attr_get_handle(ABTI_thread_attr *p_attr)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_thread_attr h_attr;
    if (p_attr == NULL) {
        h_attr = ABT_THREAD_ATTR_NULL;
    } else {
        h_attr = (ABT_thread_attr)p_attr;
    }
    return h_attr;
#else
    return (ABT_thread_attr)p_attr;
#endif
}

static inline void ABTI_thread_attr_init(ABTI_thread_attr *p_attr,
                                         void *p_stack, size_t stacksize,
                                         ABTI_thread_type thread_type,
                                         ABT_bool migratable)
{
    p_attr->p_stack = p_stack;
    p_attr->stacksize = stacksize;
    p_attr->thread_type = thread_type;
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    p_attr->migratable = migratable;
    p_attr->f_cb = NULL;
    p_attr->p_cb_arg = NULL;
#endif
}

#endif /* ABTI_THREAD_ATTR_H_INCLUDED */
