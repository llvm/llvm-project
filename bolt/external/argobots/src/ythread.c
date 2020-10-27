/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

void ABTI_ythread_set_blocked(ABTI_ythread *p_ythread)
{
    /* The root thread cannot be blocked */
    ABTI_ASSERT(!(p_ythread->thread.type & ABTI_THREAD_TYPE_ROOT));

    /* To prevent the scheduler from adding the ULT to the pool */
    ABTI_thread_set_request(&p_ythread->thread, ABTI_THREAD_REQ_BLOCK);

    /* Change the ULT's state to BLOCKED */
    ABTD_atomic_release_store_int(&p_ythread->thread.state,
                                  ABT_THREAD_STATE_BLOCKED);

    /* Increase the number of blocked ULTs */
    ABTI_pool *p_pool = p_ythread->thread.p_pool;
    ABTI_pool_inc_num_blocked(p_pool);

    LOG_DEBUG("[U%" PRIu64 ":E%d] blocked\n",
              ABTI_thread_get_id(&p_ythread->thread),
              p_ythread->thread.p_last_xstream->rank);
}

/* NOTE: This routine should be called after ABTI_ythread_set_blocked. */
void ABTI_ythread_suspend(ABTI_xstream **pp_local_xstream,
                          ABTI_ythread *p_ythread,
                          ABT_sync_event_type sync_event_type, void *p_sync)
{
    ABTI_xstream *p_local_xstream = *pp_local_xstream;
    ABTI_ASSERT(&p_ythread->thread == p_local_xstream->p_thread);
    ABTI_ASSERT(p_ythread->thread.p_last_xstream == p_local_xstream);

    /* Switch to the scheduler, i.e., suspend p_ythread  */
    LOG_DEBUG("[U%" PRIu64 ":E%d] suspended\n",
              ABTI_thread_get_id(&p_ythread->thread), p_local_xstream->rank);
    ABTI_ythread_context_switch_to_parent(pp_local_xstream, p_ythread,
                                          sync_event_type, p_sync);

    /* The suspended ULT resumes its execution from here. */
    LOG_DEBUG("[U%" PRIu64 ":E%d] resumed\n",
              ABTI_thread_get_id(&p_ythread->thread),
              p_ythread->thread.p_last_xstream->rank);
}

void ABTI_ythread_set_ready(ABTI_local *p_local, ABTI_ythread *p_ythread)
{
    /* The ULT must be in BLOCKED state. */
    ABTI_ASSERT(ABTD_atomic_acquire_load_int(&p_ythread->thread.state) ==
                ABT_THREAD_STATE_BLOCKED);

    /* We should wait until the scheduler of the blocked ULT resets the BLOCK
     * request. Otherwise, the ULT can be pushed to a pool here and be
     * scheduled by another scheduler if it is pushed to a shared pool. */
    while (ABTD_atomic_acquire_load_uint32(&p_ythread->thread.request) &
           ABTI_THREAD_REQ_BLOCK)
        ABTD_atomic_pause();

    LOG_DEBUG("[U%" PRIu64 ":E%d] set ready\n",
              ABTI_thread_get_id(&p_ythread->thread),
              p_ythread->thread.p_last_xstream->rank);

    ABTI_tool_event_ythread_resume(p_local, p_ythread,
                                   ABTI_local_get_xstream_or_null(p_local)
                                       ? ABTI_local_get_xstream(p_local)
                                             ->p_thread
                                       : NULL);
    /* p_ythread->thread.p_pool is loaded before ABTI_POOL_ADD_THREAD to keep
     * num_blocked consistent. Otherwise, other threads might pop p_ythread
     * that has been pushed in ABTI_POOL_ADD_THREAD and change
     * p_ythread->thread.p_pool by ABT_unit_set_associated_pool. */
    ABTI_pool *p_pool = p_ythread->thread.p_pool;

    /* Add the ULT to its associated pool */
    ABTI_pool_add_thread(&p_ythread->thread);

    /* Decrease the number of blocked threads */
    ABTI_pool_dec_num_blocked(p_pool);
}

ABTU_no_sanitize_address void ABTI_ythread_print_stack(ABTI_ythread *p_ythread,
                                                       FILE *p_os)
{
    void *p_stack = p_ythread->p_stack;
    size_t i, j, stacksize = p_ythread->stacksize;
    if (stacksize == 0 || p_stack == NULL) {
        /* Some threads do not have p_stack (e.g., the main thread) */
        return;
    }

    char buffer[32];
    const size_t value_width = 8;
    const int num_bytes = sizeof(buffer);

    for (i = 0; i < stacksize; i += num_bytes) {
        if (stacksize >= i + num_bytes) {
            memcpy(buffer, &((uint8_t *)p_stack)[i], num_bytes);
        } else {
            memset(buffer, 0, num_bytes);
            memcpy(buffer, &((uint8_t *)p_stack)[i], stacksize - i);
        }
        /* Print the stack address */
#if SIZEOF_VOID_P == 8
        fprintf(p_os, "%016" PRIxPTR ":",
                (uintptr_t)(&((uint8_t *)p_stack)[i]));
#elif SIZEOF_VOID_P == 4
        fprintf(p_os, "%08" PRIxPTR ":", (uintptr_t)(&((uint8_t *)p_stack)[i]));
#else
#error "unknown pointer size"
#endif
        /* Print the raw stack data */
        for (j = 0; j < num_bytes / value_width; j++) {
            if (value_width == 8) {
                uint64_t val = ((uint64_t *)buffer)[j];
                fprintf(p_os, " %016" PRIx64, val);
            } else if (value_width == 4) {
                uint32_t val = ((uint32_t *)buffer)[j];
                fprintf(p_os, " %08" PRIx32, val);
            } else if (value_width == 2) {
                uint16_t val = ((uint16_t *)buffer)[j];
                fprintf(p_os, " %04" PRIx16, val);
            } else {
                uint8_t val = ((uint8_t *)buffer)[j];
                fprintf(p_os, " %02" PRIx8, val);
            }
            if (j == (num_bytes / value_width) - 1)
                fprintf(p_os, "\n");
        }
    }
}
