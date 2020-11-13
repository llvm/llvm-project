/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTD_H_INCLUDED
#define ABTD_H_INCLUDED

#define __USE_GNU 1
#include <pthread.h>
#include "abtd_atomic.h"
#include "abtd_context.h"

/* Data Types */
typedef enum {
    ABTD_XSTREAM_CONTEXT_STATE_RUNNING,
    ABTD_XSTREAM_CONTEXT_STATE_WAITING,
    ABTD_XSTREAM_CONTEXT_STATE_REQ_JOIN,
    ABTD_XSTREAM_CONTEXT_STATE_REQ_TERMINATE,
} ABTD_xstream_context_state;
typedef struct ABTD_xstream_context {
    pthread_t native_thread;
    void *(*thread_f)(void *);
    void *p_arg;
    ABTD_xstream_context_state state;
    pthread_mutex_t state_lock;
    pthread_cond_t state_cond;
} ABTD_xstream_context;
typedef pthread_mutex_t ABTD_xstream_mutex;
#ifdef HAVE_PTHREAD_BARRIER_INIT
typedef pthread_barrier_t ABTD_xstream_barrier;
#else
typedef void *ABTD_xstream_barrier;
#endif
typedef struct ABTD_affinity_cpuset {
    size_t num_cpuids;
    int *cpuids;
} ABTD_affinity_cpuset;

/* ES Storage Qualifier */
#define ABTD_XSTREAM_LOCAL __thread

/* Environment */
void ABTD_env_init(ABTI_global *p_global);

/* ES Context */
ABTU_ret_err int ABTD_xstream_context_create(void *(*f_xstream)(void *),
                                             void *p_arg,
                                             ABTD_xstream_context *p_ctx);
void ABTD_xstream_context_free(ABTD_xstream_context *p_ctx);
void ABTD_xstream_context_join(ABTD_xstream_context *p_ctx);
void ABTD_xstream_context_revive(ABTD_xstream_context *p_ctx);
void ABTD_xstream_context_set_self(ABTD_xstream_context *p_ctx);

/* ES Affinity */
void ABTD_affinity_init(const char *affinity_str);
void ABTD_affinity_finalize(void);
ABTU_ret_err int ABTD_affinity_cpuset_read(ABTD_xstream_context *p_ctx,
                                           ABTD_affinity_cpuset *p_cpuset);
ABTU_ret_err int
ABTD_affinity_cpuset_apply(ABTD_xstream_context *p_ctx,
                           const ABTD_affinity_cpuset *p_cpuset);
int ABTD_affinity_cpuset_apply_default(ABTD_xstream_context *p_ctx, int rank);
void ABTD_affinity_cpuset_destroy(ABTD_affinity_cpuset *p_cpuset);

/* ES Affinity Parser */
typedef struct ABTD_affinity_id_list {
    int num;
    int *ids; /* id here can be negative. */
} ABTD_affinity_id_list;
typedef struct ABTD_affinity_parser_list {
    int num;
    ABTD_affinity_id_list **p_id_lists;
} ABTD_affinity_list;
ABTD_affinity_list *ABTD_affinity_list_create(const char *affinity_str);
void ABTD_affinity_list_free(ABTD_affinity_list *p_list);

#include "abtd_stream.h"

/* ULT Context */
#include "abtd_ythread.h"
void ABTD_ythread_exit(ABTI_xstream *p_local_xstream, ABTI_ythread *p_ythread);
void ABTD_ythread_cancel(ABTI_xstream *p_local_xstream,
                         ABTI_ythread *p_ythread);

#if defined(ABT_CONFIG_USE_CLOCK_GETTIME)
#include <time.h>
typedef struct timespec ABTD_time;

#elif defined(ABT_CONFIG_USE_MACH_ABSOLUTE_TIME)
#include <mach/mach_time.h>
typedef uint64_t ABTD_time;

#elif defined(ABT_CONFIG_USE_GETTIMEOFDAY)
#include <sys/time.h>
typedef struct timeval ABTD_time;

#endif

void ABTD_time_init(void);
void ABTD_time_get(ABTD_time *p_time);
double ABTD_time_read_sec(ABTD_time *p_time);

#endif /* ABTD_H_INCLUDED */
