/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_YTHREAD_HTABLE_H_INCLUDED
#define ABTI_YTHREAD_HTABLE_H_INCLUDED

#include "abt_config.h"

#if defined(HAVE_LH_LOCK_H)
#include <lh_lock.h>
#elif defined(HAVE_CLH_H)
#include <clh.h>
#else
#define USE_PTHREAD_MUTEX
#endif

struct ABTI_ythread_queue {
    ABTD_atomic_uint32 mutex; /* can be initialized by just assigning 0*/
    uint32_t num_handovers;
    uint32_t num_threads;
    uint32_t pad0;
    ABTI_ythread *head;
    ABTI_ythread *tail;
    char pad1[64 - sizeof(ABTD_atomic_uint32) - sizeof(uint32_t) * 3 -
              sizeof(ABTI_ythread *) * 2];

    /* low priority queue */
    ABTD_atomic_uint32 low_mutex; /* can be initialized by just assigning 0*/
    uint32_t low_num_threads;
    ABTI_ythread *low_head;
    ABTI_ythread *low_tail;
    char pad2[64 - sizeof(ABTD_atomic_uint32) - sizeof(uint32_t) -
              sizeof(ABTI_ythread *) * 2];

    /* two doubly-linked lists */
    ABTI_ythread_queue *p_h_next;
    ABTI_ythread_queue *p_h_prev;
    ABTI_ythread_queue *p_l_next;
    ABTI_ythread_queue *p_l_prev;
    char pad3[64 - sizeof(ABTI_ythread_queue *) * 4];
};

struct ABTI_ythread_htable {
#if defined(HAVE_LH_LOCK_H)
    lh_lock_t mutex;
#elif defined(HAVE_CLH_H)
    clh_lock_t mutex;
#elif defined(USE_PTHREAD_MUTEX)
    pthread_mutex_t mutex;
#else
    ABTI_spinlock mutex; /* To protect table */
#endif
    ABTD_atomic_uint32 num_elems;
    uint32_t num_rows;
    ABTI_ythread_queue *queue;

    ABTI_ythread_queue *h_list; /* list of non-empty high prio. queues */
    ABTI_ythread_queue *l_list; /* list of non-empty low prio. queues */
};

#if defined(HAVE_LH_LOCK_H)
#define ABTI_THREAD_HTABLE_LOCK(m) lh_acquire_lock(&m)
#define ABTI_THREAD_HTABLE_UNLOCK(m) lh_release_lock(&m)
#elif defined(HAVE_CLH_H)
#define ABTI_THREAD_HTABLE_LOCK(m) clh_acquire(&m)
#define ABTI_THREAD_HTABLE_UNLOCK(m) clh_release(&m)
#elif defined(USE_PTHREAD_MUTEX)
#define ABTI_THREAD_HTABLE_LOCK(m) pthread_mutex_lock(&m)
#define ABTI_THREAD_HTABLE_UNLOCK(m) pthread_mutex_unlock(&m)
#else
#define ABTI_THREAD_HTABLE_LOCK(m) ABTI_spinlock_acquire(&m)
#define ABTI_THREAD_HTABLE_UNLOCK(m) ABTI_spinlock_release(&m)
#endif

static inline void ABTI_ythread_queue_acquire_mutex(ABTI_ythread_queue *p_queue)
{
    while (!ABTD_atomic_bool_cas_weak_uint32(&p_queue->mutex, 0, 1)) {
        while (ABTD_atomic_acquire_load_uint32(&p_queue->mutex) != 0)
            ;
    }
}

static inline void ABTI_ythread_queue_release_mutex(ABTI_ythread_queue *p_queue)
{
    ABTD_atomic_release_store_uint32(&p_queue->mutex, 0);
}

static inline void
ABTI_ythread_queue_acquire_low_mutex(ABTI_ythread_queue *p_queue)
{
    while (!ABTD_atomic_bool_cas_weak_uint32(&p_queue->low_mutex, 0, 1)) {
        while (ABTD_atomic_acquire_load_uint32(&p_queue->low_mutex) != 0)
            ;
    }
}

static inline void
ABTI_ythread_queue_release_low_mutex(ABTI_ythread_queue *p_queue)
{
    ABTD_atomic_release_store_uint32(&p_queue->low_mutex, 0);
}

static inline void ABTI_ythread_htable_add_h_node(ABTI_ythread_htable *p_htable,
                                                  ABTI_ythread_queue *p_node)
{
    ABTI_ythread_queue *p_curr = p_htable->h_list;
    if (!p_curr) {
        p_node->p_h_next = p_node;
        p_node->p_h_prev = p_node;
        p_htable->h_list = p_node;
    } else if (!p_node->p_h_next) {
        p_node->p_h_next = p_curr;
        p_node->p_h_prev = p_curr->p_h_prev;
        p_curr->p_h_prev->p_h_next = p_node;
        p_curr->p_h_prev = p_node;
    }
}

static inline void ABTI_ythread_htable_del_h_head(ABTI_ythread_htable *p_htable)
{
    ABTI_ythread_queue *p_prev, *p_next;
    ABTI_ythread_queue *p_node = p_htable->h_list;

    if (p_node == p_node->p_h_next) {
        p_node->p_h_next = NULL;
        p_node->p_h_prev = NULL;
        p_htable->h_list = NULL;
    } else {
        p_prev = p_node->p_h_prev;
        p_next = p_node->p_h_next;
        p_prev->p_h_next = p_next;
        p_next->p_h_prev = p_prev;
        p_node->p_h_next = NULL;
        p_node->p_h_prev = NULL;
        p_htable->h_list = p_next;
    }
}

static inline void ABTI_ythread_htable_add_l_node(ABTI_ythread_htable *p_htable,
                                                  ABTI_ythread_queue *p_node)
{
    ABTI_ythread_queue *p_curr = p_htable->l_list;
    if (!p_curr) {
        p_node->p_l_next = p_node;
        p_node->p_l_prev = p_node;
        p_htable->l_list = p_node;
    } else if (!p_node->p_l_next) {
        p_node->p_l_next = p_curr;
        p_node->p_l_prev = p_curr->p_l_prev;
        p_curr->p_l_prev->p_l_next = p_node;
        p_curr->p_l_prev = p_node;
    }
}

static inline void ABTI_ythread_htable_del_l_head(ABTI_ythread_htable *p_htable)
{
    ABTI_ythread_queue *p_prev, *p_next;
    ABTI_ythread_queue *p_node = p_htable->l_list;

    if (p_node == p_node->p_l_next) {
        p_node->p_l_next = NULL;
        p_node->p_l_prev = NULL;
        p_htable->l_list = NULL;
    } else {
        p_prev = p_node->p_l_prev;
        p_next = p_node->p_l_next;
        p_prev->p_l_next = p_next;
        p_next->p_l_prev = p_prev;
        p_node->p_l_next = NULL;
        p_node->p_l_prev = NULL;
        p_htable->l_list = p_next;
    }
}

#endif /* ABTI_YTHREAD_HTABLE_H_INCLUDED */
