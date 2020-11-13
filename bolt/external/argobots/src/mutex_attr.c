/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

/** @defgroup MUTEX_ATTR Mutex Attributes
 * Attributes are used to specify mutex behavior that is different from the
 * default.  When a mutex is created with \c ABT_mutex_create_with_attr(),
 * attributes can be specified with an \c ABT_mutex_attr object.
 */

/**
 * @ingroup MUTEX_ATTR
 * @brief   Create a new mutex attribute object.
 *
 * \c ABT_mutex_attr_create() creates a mutex attribute object with default
 * attribute values.  The handle to the attribute object is returned through
 * \c newattr. The attribute object can be used in more than one mutex.
 *
 * @param[out] newattr  handle to a new attribute object
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_mutex_attr_create(ABT_mutex_attr *newattr)
{
    ABTI_mutex_attr *p_newattr;

    int abt_errno = ABTU_malloc(sizeof(ABTI_mutex_attr), (void **)&p_newattr);
    ABTI_CHECK_ERROR(abt_errno);

    /* Default values */
    p_newattr->attrs = ABTI_MUTEX_ATTR_NONE;
    p_newattr->nesting_cnt = 0;
    p_newattr->owner_id = 0;
    p_newattr->max_handovers = gp_ABTI_global->mutex_max_handovers;
    p_newattr->max_wakeups = gp_ABTI_global->mutex_max_wakeups;

    /* Return value */
    *newattr = ABTI_mutex_attr_get_handle(p_newattr);
    return ABT_SUCCESS;
}

/**
 * @ingroup MUTEX_ATTR
 * @brief   Free the mutex attribute object.
 *
 * \c ABT_mutex_attr_free() deallocates memory used for the mutex attribute
 * object.  If this function successfully returns, \c attr will be set to
 * \c ABT_MUTEX_ATTR_NULL.
 *
 * @param[in,out] attr  handle to the target attribute object
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_mutex_attr_free(ABT_mutex_attr *attr)
{
    ABT_mutex_attr h_attr = *attr;
    ABTI_mutex_attr *p_attr = ABTI_mutex_attr_get_ptr(h_attr);
    ABTI_CHECK_NULL_MUTEX_ATTR_PTR(p_attr);

    /* Free the memory */
    ABTU_free(p_attr);
    /* Return value */
    *attr = ABT_MUTEX_ATTR_NULL;
    return ABT_SUCCESS;
}

/**
 * @ingroup MUTEX_ATTR
 * @brief   Set the recursive property in the attribute object.
 *
 * \c ABT_mutex_attr_set_recursive() sets the recursive property (i.e., whether
 * the mutex can be locked multiple times by the same owner) in the attribute
 * object associated with handle \c attr.
 *
 * @param[in] attr       handle to the target attribute object
 * @param[in] recursive  boolean value for the recursive locking support
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_mutex_attr_set_recursive(ABT_mutex_attr attr, ABT_bool recursive)
{
    ABTI_mutex_attr *p_attr = ABTI_mutex_attr_get_ptr(attr);
    ABTI_CHECK_NULL_MUTEX_ATTR_PTR(p_attr);

    /* Set the value */
    if (recursive == ABT_TRUE) {
        p_attr->attrs |= ABTI_MUTEX_ATTR_RECURSIVE;
    } else {
        p_attr->attrs &= ~ABTI_MUTEX_ATTR_RECURSIVE;
    }
    return ABT_SUCCESS;
}
