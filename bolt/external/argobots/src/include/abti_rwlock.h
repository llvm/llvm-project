/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_RWLOCK_H_INCLUDED
#define ABTI_RWLOCK_H_INCLUDED

#include "abti_mutex.h"
#include "abti_cond.h"

/* Inlined functions for RWLock */

static inline ABTI_rwlock *ABTI_rwlock_get_ptr(ABT_rwlock rwlock)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABTI_rwlock *p_rwlock;
    if (rwlock == ABT_RWLOCK_NULL) {
        p_rwlock = NULL;
    } else {
        p_rwlock = (ABTI_rwlock *)rwlock;
    }
    return p_rwlock;
#else
    return (ABTI_rwlock *)rwlock;
#endif
}

static inline ABT_rwlock ABTI_rwlock_get_handle(ABTI_rwlock *p_rwlock)
{
#ifndef ABT_CONFIG_DISABLE_ERROR_CHECK
    ABT_rwlock h_rwlock;
    if (p_rwlock == NULL) {
        h_rwlock = ABT_RWLOCK_NULL;
    } else {
        h_rwlock = (ABT_rwlock)p_rwlock;
    }
    return h_rwlock;
#else
    return (ABT_rwlock)p_rwlock;
#endif
}

#endif /* ABTI_RWLOCK_H_INCLUDED */
