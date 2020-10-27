/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <stdio.h>
#include <stdarg.h>
#include "abti.h"

#ifdef ABT_CONFIG_USE_DEBUG_LOG

void ABTI_log_debug(FILE *fh, const char *format, ...)
{
    if (gp_ABTI_global->use_logging == ABT_FALSE)
        return;
    ABTI_local *p_local = ABTI_local_get_local_uninlined();

    const char *prefix_fmt = NULL, *prefix = NULL;
    char *newfmt;
    uint64_t tid;
    int rank;
    size_t newfmt_len;

    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
    if (p_local_xstream) {
        ABTI_ythread *p_ythread =
            ABTI_thread_get_ythread(p_local_xstream->p_thread);
        if (p_ythread == NULL) {
            if (p_local_xstream->type != ABTI_XSTREAM_TYPE_PRIMARY) {
                prefix_fmt = "<U%" PRIu64 ":E%d> %s";
                rank = p_local_xstream->rank;
                tid = 0;
            } else {
                prefix = "<U0:E0> ";
                prefix_fmt = "%s%s";
            }
        } else {
            rank = p_local_xstream->rank;
            prefix_fmt = "<U%" PRIu64 ":E%d> %s";
            tid = ABTI_thread_get_id(&p_ythread->thread);
        }
    } else {
        prefix = "<EXT> ";
        prefix_fmt = "%s%s";
    }

    if (prefix == NULL) {
        /* Both tid and rank are less than 42 characters in total. */
        const int len_tid_rank = 50;
        newfmt_len = 6 + len_tid_rank + strlen(format);
        int abt_errno = ABTU_malloc(newfmt_len + 1, (void **)&newfmt);
        if (abt_errno != ABT_SUCCESS)
            return;
        sprintf(newfmt, prefix_fmt, tid, rank, format);
    } else {
        newfmt_len = strlen(prefix) + strlen(format);
        int abt_errno = ABTU_malloc(newfmt_len + 1, (void **)&newfmt);
        if (abt_errno != ABT_SUCCESS)
            return;
        sprintf(newfmt, prefix_fmt, prefix, format);
    }

#ifndef ABT_CONFIG_USE_DEBUG_LOG_DISCARD
    va_list list;
    va_start(list, format);
    vfprintf(fh, newfmt, list);
    va_end(list);
    fflush(fh);
#else
    /* Discard the log message.  This option is used to check if the logging
     * function works correct (i.e., without any SEGV) but a tester does not
     * need an actual log since the output can be extremely large. */
#endif

    ABTU_free(newfmt);
}

void ABTI_log_pool_push(ABTI_pool *p_pool, ABT_unit unit)
{
    if (gp_ABTI_global->use_logging == ABT_FALSE)
        return;

    ABTI_ythread *p_ythread = NULL;
    ABTI_thread *p_task = NULL;
    switch (p_pool->u_get_type(unit)) {
        case ABT_UNIT_TYPE_THREAD:
            p_ythread = ABTI_ythread_get_ptr(p_pool->u_get_thread(unit));
            if (p_ythread->thread.p_last_xstream) {
                LOG_DEBUG("[U%" PRIu64 ":E%d] pushed to P%" PRIu64 "\n",
                          ABTI_thread_get_id(&p_ythread->thread),
                          p_ythread->thread.p_last_xstream->rank, p_pool->id);
            } else {
                LOG_DEBUG("[U%" PRIu64 "] pushed to P%" PRIu64 "\n",
                          ABTI_thread_get_id(&p_ythread->thread), p_pool->id);
            }
            break;

        case ABT_UNIT_TYPE_TASK:
            p_task = ABTI_thread_get_ptr(p_pool->u_get_task(unit));
            if (p_task->p_last_xstream) {
                LOG_DEBUG("[T%" PRIu64 ":E%d] pushed to P%" PRIu64 "\n",
                          ABTI_thread_get_id(p_task),
                          p_task->p_last_xstream->rank, p_pool->id);
            } else {
                LOG_DEBUG("[T%" PRIu64 "] pushed to P%" PRIu64 "\n",
                          ABTI_thread_get_id(p_task), p_pool->id);
            }
            break;

        default:
            ABTI_ASSERT(0);
            ABTU_unreachable();
    }
}

void ABTI_log_pool_remove(ABTI_pool *p_pool, ABT_unit unit)
{
    if (gp_ABTI_global->use_logging == ABT_FALSE)
        return;

    ABTI_ythread *p_ythread = NULL;
    ABTI_thread *p_task = NULL;
    switch (p_pool->u_get_type(unit)) {
        case ABT_UNIT_TYPE_THREAD:
            p_ythread = ABTI_ythread_get_ptr(p_pool->u_get_thread(unit));
            if (p_ythread->thread.p_last_xstream) {
                LOG_DEBUG("[U%" PRIu64 ":E%d] removed from P%" PRIu64 "\n",
                          ABTI_thread_get_id(&p_ythread->thread),
                          p_ythread->thread.p_last_xstream->rank, p_pool->id);
            } else {
                LOG_DEBUG("[U%" PRIu64 "] removed from P%" PRIu64 "\n",
                          ABTI_thread_get_id(&p_ythread->thread), p_pool->id);
            }
            break;

        case ABT_UNIT_TYPE_TASK:
            p_task = ABTI_thread_get_ptr(p_pool->u_get_task(unit));
            if (p_task->p_last_xstream) {
                LOG_DEBUG("[T%" PRIu64 ":E%d] removed from P%" PRIu64 "\n",
                          ABTI_thread_get_id(p_task),
                          p_task->p_last_xstream->rank, p_pool->id);
            } else {
                LOG_DEBUG("[T%" PRIu64 "] removed from P%" PRIu64 "\n",
                          ABTI_thread_get_id(p_task), p_pool->id);
            }
            break;

        default:
            ABTI_ASSERT(0);
            ABTU_unreachable();
    }
}

void ABTI_log_pool_pop(ABTI_pool *p_pool, ABT_unit unit)
{
    if (gp_ABTI_global->use_logging == ABT_FALSE)
        return;
    if (unit == ABT_UNIT_NULL)
        return;

    ABTI_ythread *p_ythread = NULL;
    ABTI_thread *p_task = NULL;
    switch (p_pool->u_get_type(unit)) {
        case ABT_UNIT_TYPE_THREAD:
            p_ythread = ABTI_ythread_get_ptr(p_pool->u_get_thread(unit));
            if (p_ythread->thread.p_last_xstream) {
                LOG_DEBUG("[U%" PRIu64 ":E%d] popped from "
                          "P%" PRIu64 "\n",
                          ABTI_thread_get_id(&p_ythread->thread),
                          p_ythread->thread.p_last_xstream->rank, p_pool->id);
            } else {
                LOG_DEBUG("[U%" PRIu64 "] popped from P%" PRIu64 "\n",
                          ABTI_thread_get_id(&p_ythread->thread), p_pool->id);
            }
            break;

        case ABT_UNIT_TYPE_TASK:
            p_task = ABTI_thread_get_ptr(p_pool->u_get_task(unit));
            if (p_task->p_last_xstream) {
                LOG_DEBUG("[T%" PRIu64 ":E%d] popped from "
                          "P%" PRIu64 "\n",
                          ABTI_thread_get_id(p_task),
                          p_task->p_last_xstream->rank, p_pool->id);
            } else {
                LOG_DEBUG("[T%" PRIu64 "] popped from P%" PRIu64 "\n",
                          ABTI_thread_get_id(p_task), p_pool->id);
            }
            break;

        default:
            ABTI_ASSERT(0);
            ABTU_unreachable();
    }
}

#endif /* ABT_CONFIG_USE_DEBUG_LOG */
