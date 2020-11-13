/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

/** @defgroup ULT_ATTR ULT Attributes
 * Attributes are used to specify ULT behavior that is different from the
 * default. When a ULT is created with \c ABT_thread_create(), attributes
 * can be specified with an \c ABT_thread_attr object.
 */

/**
 * @ingroup ULT_ATTR
 * @brief   Create a new ULT attribute object.
 *
 * \c ABT_thread_attr_create() creates a ULT attribute object with default
 * attribute values. The handle to the attribute object is returned through
 * \c newattr. The attribute object can be used in more than one ULT.
 *
 * @param[out] newattr  handle to a new attribute object
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_attr_create(ABT_thread_attr *newattr)
{
    ABTI_thread_attr *p_newattr;
    int abt_errno = ABTU_malloc(sizeof(ABTI_thread_attr), (void **)&p_newattr);
    ABTI_CHECK_ERROR(abt_errno);

    /* Default values */
    ABTI_thread_attr_init(p_newattr, NULL, gp_ABTI_global->thread_stacksize,
                          ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC_STACK, ABT_TRUE);
    *newattr = ABTI_thread_attr_get_handle(p_newattr);
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT_ATTR
 * @brief   Free the ULT attribute object.
 *
 * \c ABT_thread_attr_free() deallocates memory used for the ULT attribute
 * object. If this function successfully returns, \c attr will be set to
 * \c ABT_THREAD_ATTR_NULL.
 *
 * @param[in,out] attr  handle to the target attribute object
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_attr_free(ABT_thread_attr *attr)
{
    ABT_thread_attr h_attr = *attr;
    ABTI_thread_attr *p_attr = ABTI_thread_attr_get_ptr(h_attr);
    ABTI_CHECK_NULL_THREAD_ATTR_PTR(p_attr);

    /* Free the memory */
    ABTU_free(p_attr);
    *attr = ABT_THREAD_ATTR_NULL;
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT_ATTR
 * @brief   Set stack attributes.
 *
 * \c ABT_thread_attr_set_stack() sets the stack address and the stack size
 * (in bytes) in the attribute object associated with handle \c attr.
 * If \c attr is used to create a ULT, the memory pointed to by \c stackaddr
 * will be used as the stack area for the new ULT.
 *
 * If \c stackaddr is \c NULL, a stack with size \c stacksize will be created
 * by the Argobots runtime.  If it is not \c NULL, it should be aligned by 8
 * (i.e., \c stackaddr & 0x7 must be zero), and the user has to deallocate
 * the stack memory after the ULT, for which \c attr was used, terminates.
 *
 * @param[in] attr       handle to the target attribute object
 * @param[in] stackaddr  stack address
 * @param[in] stacksize  stack size in bytes
 * @return Error code
 * @retval ABT_SUCCESS   on success
 * @retval ABT_ERR_OTHER invalid stack address
 */
int ABT_thread_attr_set_stack(ABT_thread_attr attr, void *stackaddr,
                              size_t stacksize)
{
    ABTI_thread_attr *p_attr = ABTI_thread_attr_get_ptr(attr);
    ABTI_CHECK_NULL_THREAD_ATTR_PTR(p_attr);

    ABTI_thread_type new_thread_type;
    if (stackaddr != NULL) {
        if (((uintptr_t)stackaddr & 0x7) != 0) {
            ABTI_HANDLE_ERROR(ABT_ERR_OTHER);
        }
        new_thread_type = ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC;
    } else {
        if (stacksize == gp_ABTI_global->thread_stacksize) {
            new_thread_type = ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC_STACK;
        } else {
            new_thread_type = ABTI_THREAD_TYPE_MEM_MALLOC_DESC_STACK;
        }
    }
    /* Unset the stack type and set new_thread_type. */
    p_attr->thread_type &= ~ABTI_THREAD_TYPES_MEM;
    p_attr->thread_type |= new_thread_type;

    p_attr->p_stack = stackaddr;
    p_attr->stacksize = stacksize;
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT_ATTR
 * @brief   Get stack attributes.
 *
 * \c ABT_thread_attr_get_stack() retrieves the stack address and the stack
 * size (in bytes) from the attribute \c attr to \c stackaddr and \c stacksize,
 * respectively.
 *
 * The user can obtain the ULT's attributes using \c ABT_thread_get_attr().
 *
 * @param[in]  attr       handle to the target attribute object
 * @param[out] stackaddr  stack address
 * @param[out] stacksize  stack size in bytes
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_attr_get_stack(ABT_thread_attr attr, void **stackaddr,
                              size_t *stacksize)
{
    ABTI_thread_attr *p_attr = ABTI_thread_attr_get_ptr(attr);
    ABTI_CHECK_NULL_THREAD_ATTR_PTR(p_attr);

    *stackaddr = p_attr->p_stack;
    *stacksize = p_attr->stacksize;
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT_ATTR
 * @brief   Set the stack size in the attribute object.
 *
 * \c ABT_thread_attr_set_stacksize() sets the stack size (in bytes) in the
 * attribute object associated with handle \c attr.
 *
 * @param[in] attr       handle to the target attribute object
 * @param[in] stacksize  stack size in bytes
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_attr_set_stacksize(ABT_thread_attr attr, size_t stacksize)
{
    ABTI_thread_attr *p_attr = ABTI_thread_attr_get_ptr(attr);
    ABTI_CHECK_NULL_THREAD_ATTR_PTR(p_attr);

    /* Set the value */
    p_attr->stacksize = stacksize;
    ABTI_thread_type new_thread_type;
    if (stacksize == gp_ABTI_global->thread_stacksize) {
        new_thread_type = ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC_STACK;
    } else {
        new_thread_type = ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC_STACK;
    }
    /* Unset the stack type and set new_thread_type. */
    p_attr->thread_type &= ~ABTI_THREAD_TYPES_MEM;
    p_attr->thread_type |= new_thread_type;
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT_ATTR
 * @brief   Get the stack size from the attribute object.
 *
 * \c ABT_thread_attr_get_stacksize() returns the stack size (in bytes) through
 * \c stacksize from the attribute object associated with handle \c attr.
 *
 * @param[in]  attr       handle to the target attribute object
 * @param[out] stacksize  stack size in bytes
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_attr_get_stacksize(ABT_thread_attr attr, size_t *stacksize)
{
    ABTI_thread_attr *p_attr = ABTI_thread_attr_get_ptr(attr);
    ABTI_CHECK_NULL_THREAD_ATTR_PTR(p_attr);

    *stacksize = p_attr->stacksize;
    return ABT_SUCCESS;
}

/**
 * @ingroup ULT_ATTR
 * @brief   Set callback function and its argument in the attribute object.
 *
 * \c ABT_thread_attr_set_callback() sets the callback function and its
 * argument, which will be invoked on ULT migration.
 *
 * @param[in] attr     handle to the target attribute object
 * @param[in] cb_func  callback function pointer
 * @param[in] cb_arg   argument for the callback function
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_attr_set_callback(ABT_thread_attr attr,
                                 void (*cb_func)(ABT_thread thread,
                                                 void *cb_arg),
                                 void *cb_arg)
{
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    ABTI_thread_attr *p_attr = ABTI_thread_attr_get_ptr(attr);
    ABTI_CHECK_NULL_THREAD_ATTR_PTR(p_attr);

    /* Set the value */
    p_attr->f_cb = cb_func;
    p_attr->p_cb_arg = cb_arg;
    return ABT_SUCCESS;
#else
    ABTI_HANDLE_ERROR(ABT_ERR_FEATURE_NA);
#endif
}

/**
 * @ingroup ULT_ATTR
 * @brief   Set the ULT's migratability in the attribute object.
 *
 * \c ABT_thread_attr_set_migratable() sets the ULT's migratability in the
 * target attribute object.
 * If \c flag is \c ABT_TRUE, the ULT created with this attribute becomes
 * migratable. On the other hand, if \ flag is \c ABT_FALSE, the ULT created
 * with this attribute becomes unmigratable.
 *
 * @param[in] attr  handle to the target attribute object
 * @param[in] flag  migratability flag (<tt>ABT_TRUE</tt>: migratable,
 *                  <tt>ABT_FALSE</tt>: not)
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_thread_attr_set_migratable(ABT_thread_attr attr, ABT_bool flag)
{
#ifndef ABT_CONFIG_DISABLE_MIGRATION
    ABTI_thread_attr *p_attr = ABTI_thread_attr_get_ptr(attr);
    ABTI_CHECK_NULL_THREAD_ATTR_PTR(p_attr);

    /* Set the value */
    p_attr->migratable = flag;
    return ABT_SUCCESS;
#else
    ABTI_HANDLE_ERROR(ABT_ERR_FEATURE_NA);
#endif
}

/*****************************************************************************/
/* Private APIs                                                              */
/*****************************************************************************/

void ABTI_thread_attr_print(ABTI_thread_attr *p_attr, FILE *p_os, int indent)
{
    if (p_attr == NULL) {
        fprintf(p_os, "%*sULT attr: [NULL ATTR]\n", indent, "");
    } else {
        const char *stacktype;
        if (p_attr->thread_type & ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC) {
            stacktype = "MEMPOOL_DESC";
        } else if (p_attr->thread_type & ABTI_THREAD_TYPE_MEM_MALLOC_DESC) {
            stacktype = "MALLOC_DESC";
        } else if (p_attr->thread_type &
                   ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC_STACK) {
            stacktype = "MEMPOOL_DESC_STACK";
        } else if (p_attr->thread_type &
                   ABTI_THREAD_TYPE_MEM_MALLOC_DESC_STACK) {
            stacktype = "MALLOC_DESC_STACK";
        } else {
            stacktype = "UNKNOWN";
        }
#ifndef ABT_CONFIG_DISABLE_MIGRATION
        fprintf(p_os,
                "%*sULT attr: ["
                "stack:%p "
                "stacksize:%zu "
                "stacktype:%s "
                "migratable:%s "
                "cb_arg:%p"
                "]\n",
                indent, "", p_attr->p_stack, p_attr->stacksize, stacktype,
                (p_attr->migratable == ABT_TRUE ? "TRUE" : "FALSE"),
                p_attr->p_cb_arg);
#else
        fprintf(p_os,
                "%*sULT attr: ["
                "stack:%p "
                "stacksize:%zu "
                "stacktype:%s "
                "]\n",
                indent, "", p_attr->p_stack, p_attr->stacksize, stacktype);
#endif
    }
    fflush(p_os);
}

ABTU_ret_err int ABTI_thread_attr_dup(const ABTI_thread_attr *p_attr,
                                      ABTI_thread_attr **pp_dup_attr)
{
    ABTI_thread_attr *p_dup_attr;
    int abt_errno = ABTU_malloc(sizeof(ABTI_thread_attr), (void **)&p_dup_attr);
    ABTI_CHECK_ERROR(abt_errno);

    memcpy(p_dup_attr, p_attr, sizeof(ABTI_thread_attr));
    *pp_dup_attr = p_dup_attr;
    return ABT_SUCCESS;
}
