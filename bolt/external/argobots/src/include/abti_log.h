/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_LOG_H_INCLUDED
#define ABTI_LOG_H_INCLUDED

#include "abt_config.h"

#ifdef ABT_CONFIG_USE_DEBUG_LOG

void ABTI_log_debug(FILE *fh, const char *format, ...);
void ABTI_log_pool_push(ABTI_pool *p_pool, ABT_unit unit);
void ABTI_log_pool_remove(ABTI_pool *p_pool, ABT_unit unit);
void ABTI_log_pool_pop(ABTI_pool *p_pool, ABT_unit unit);

#define LOG_DEBUG(fmt, ...) ABTI_log_debug(stderr, fmt, __VA_ARGS__)

#define LOG_DEBUG_POOL_PUSH(p_pool, unit) ABTI_log_pool_push(p_pool, unit)
#define LOG_DEBUG_POOL_REMOVE(p_pool, unit) ABTI_log_pool_remove(p_pool, unit)
#define LOG_DEBUG_POOL_POP(p_pool, unit) ABTI_log_pool_pop(p_pool, unit)

#else

#define LOG_DEBUG(fmt, ...)

#define LOG_DEBUG_POOL_PUSH(p_pool, unit)
#define LOG_DEBUG_POOL_REMOVE(p_pool, unit)
#define LOG_DEBUG_POOL_POP(p_pool, unit)

#endif /* ABT_CONFIG_USE_DEBUG_LOG */

#endif /* ABTI_LOG_H_INCLUDED */
