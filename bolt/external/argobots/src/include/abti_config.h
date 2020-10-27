/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_CONFIG_H_INCLUDED
#define ABTI_CONFIG_H_INCLUDED

/* Inlined functions for Config */

static inline ABTI_sched_config *
ABTI_sched_config_get_ptr(ABT_sched_config config)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_sched_config *p_config;
    if (config == ABT_SCHED_CONFIG_NULL) {
        p_config = NULL;
    } else {
        p_config = (ABTI_sched_config *)config;
    }
    return p_config;
#else
    return (ABTI_sched_config *)config;
#endif
}

static inline ABT_sched_config
ABTI_sched_config_get_handle(ABTI_sched_config *p_config)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_sched_config h_config;
    if (p_config == NULL) {
        h_config = ABT_SCHED_CONFIG_NULL;
    } else {
        h_config = (ABT_sched_config)p_config;
    }
    return h_config;
#else
    return (ABT_sched_config)p_config;
#endif
}

#endif /* ABTI_CONFIG_H_INCLUDED */
