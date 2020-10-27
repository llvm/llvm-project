/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

/** @defgroup UNIT  Work Unit
 * This group is for work units.
 */

/**
 * @ingroup UNIT
 * @brief   Set the associated pool for the target work unit.
 *
 * \c ABT_unit_set_associated_pool() changes the associated pool of the target
 * work unit \c unit, such as ULT or tasklet, to \c pool.  This routine must be
 * called after \c unit is popped from its original associated pool (i.e.,
 * \c unit must not be inside any pool), which is the pool where \c unit was
 * residing in.
 *
 * @param[in] unit  handle to the work unit
 * @param[in] pool  handle to the pool
 * @return Error code
 * @retval ABT_SUCCESS on success
 */
int ABT_unit_set_associated_pool(ABT_unit unit, ABT_pool pool)
{
    ABTI_pool *p_pool = ABTI_pool_get_ptr(pool);
    ABTI_CHECK_NULL_POOL_PTR(p_pool);

    ABTI_unit_set_associated_pool(unit, p_pool);
    return ABT_SUCCESS;
}

/*****************************************************************************/
/* Private APIs                                                              */
/*****************************************************************************/

void ABTI_unit_set_associated_pool(ABT_unit unit, ABTI_pool *p_pool)
{
    ABT_unit_type type = p_pool->u_get_type(unit);

    if (type == ABT_UNIT_TYPE_THREAD) {
        ABT_thread thread = p_pool->u_get_thread(unit);
        ABTI_ythread *p_thread = ABTI_ythread_get_ptr(thread);
        p_thread->thread.p_pool = p_pool;

    } else {
        ABTI_ASSERT(type == ABT_UNIT_TYPE_TASK);
        ABT_task task = p_pool->u_get_task(unit);
        ABTI_thread *p_task = ABTI_thread_get_ptr(task);
        p_task->p_pool = p_pool;
    }
}
