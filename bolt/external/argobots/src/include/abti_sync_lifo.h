/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_SYNC_LIFO_H_INCLUDED
#define ABTI_SYNC_LIFO_H_INCLUDED

/*
 * This header implements a list-based LIFO.  The implementation is lock-free if
 * architectures that support atomic operations for two consecutive pointers
 * (e.g., 128-bit CAS on 64-bit systems), which is required to avoid the ABA
 * problem.  If not, it will use a simple lock and thus become blocking.
 */

typedef struct ABTI_sync_lifo_element {
    struct ABTI_sync_lifo_element *p_next;
} ABTI_sync_lifo_element;

typedef __attribute__((aligned(ABT_CONFIG_STATIC_CACHELINE_SIZE))) struct {
#if ABTD_ATOMIC_SUPPORT_TAGGED_PTR
    ABTD_atomic_tagged_ptr p_top;
#else
    ABTI_spinlock lock;
    ABTD_atomic_ptr p_top;
#endif
} ABTI_sync_lifo;

static inline void ABTI_sync_lifo_init(ABTI_sync_lifo *p_lifo)
{
    ABTI_ASSERT(p_lifo);
#if ABTD_ATOMIC_SUPPORT_TAGGED_PTR
    ABTD_atomic_release_store_non_atomic_tagged_ptr(&p_lifo->p_top, NULL, 0);
#else
    ABTI_spinlock_clear(&p_lifo->lock);
    ABTD_atomic_relaxed_store_ptr(&p_lifo->p_top, NULL);
#endif
}

static inline void ABTI_sync_lifo_destroy(ABTI_sync_lifo *p_lifo)
{
    ; /* Do nothing. */
}

static inline void ABTI_sync_lifo_push_unsafe(ABTI_sync_lifo *p_lifo,
                                              ABTI_sync_lifo_element *p_elem)
{
#if ABTD_ATOMIC_SUPPORT_TAGGED_PTR
    ABTI_sync_lifo_element *p_cur_top;
    size_t cur_tag;
    ABTD_atomic_relaxed_load_non_atomic_tagged_ptr(&p_lifo->p_top,
                                                   (void **)&p_cur_top,
                                                   &cur_tag);
    p_elem->p_next = p_cur_top;
    ABTD_atomic_relaxed_store_non_atomic_tagged_ptr(&p_lifo->p_top, p_elem,
                                                    cur_tag + 1);
#else
    ABTI_sync_lifo_element *p_cur_top =
        (ABTI_sync_lifo_element *)ABTD_atomic_relaxed_load_ptr(&p_lifo->p_top);
    p_elem->p_next = p_cur_top;
    ABTD_atomic_relaxed_store_ptr(&p_lifo->p_top, p_elem);
#endif
}

static inline ABTI_sync_lifo_element *
ABTI_sync_lifo_pop_unsafe(ABTI_sync_lifo *p_lifo)
{
#if ABTD_ATOMIC_SUPPORT_TAGGED_PTR
    ABTI_sync_lifo_element *p_cur_top;
    size_t cur_tag;
    ABTD_atomic_relaxed_load_non_atomic_tagged_ptr(&p_lifo->p_top,
                                                   (void **)&p_cur_top,
                                                   &cur_tag);
    if (p_cur_top == NULL)
        return NULL;
    ABTI_sync_lifo_element *p_next = p_cur_top->p_next;
    ABTD_atomic_relaxed_store_non_atomic_tagged_ptr(&p_lifo->p_top, p_next,
                                                    cur_tag + 1);
    return p_cur_top;
#else
    ABTI_sync_lifo_element *p_cur_top =
        (ABTI_sync_lifo_element *)ABTD_atomic_relaxed_load_ptr(&p_lifo->p_top);
    if (!p_cur_top)
        return NULL;
    ABTD_atomic_relaxed_store_ptr(&p_lifo->p_top, p_cur_top->p_next);
    return p_cur_top;
#endif
}

static inline void ABTI_sync_lifo_push(ABTI_sync_lifo *p_lifo,
                                       ABTI_sync_lifo_element *p_elem)
{
#if ABTD_ATOMIC_SUPPORT_TAGGED_PTR
    while (1) {
        ABTI_sync_lifo_element *p_cur_top;
        size_t cur_tag;
        ABTD_atomic_acquire_load_non_atomic_tagged_ptr(&p_lifo->p_top,
                                                       (void **)&p_cur_top,
                                                       &cur_tag);
        p_elem->p_next = p_cur_top;
        /* tag is incremented to avoid the ABA problem. */
        if (ABTU_likely(ABTD_atomic_bool_cas_weak_tagged_ptr(&p_lifo->p_top,
                                                             p_cur_top, cur_tag,
                                                             p_elem,
                                                             cur_tag + 1))) {
            return;
        }
    }
#else
    ABTI_spinlock_acquire(&p_lifo->lock);
    ABTI_sync_lifo_push_unsafe(p_lifo, p_elem);
    ABTI_spinlock_release(&p_lifo->lock);
#endif
}

static inline ABTI_sync_lifo_element *ABTI_sync_lifo_pop(ABTI_sync_lifo *p_lifo)
{
#if ABTD_ATOMIC_SUPPORT_TAGGED_PTR
    while (1) {
        ABTI_sync_lifo_element *p_cur_top;
        size_t cur_tag;
        ABTD_atomic_acquire_load_non_atomic_tagged_ptr(&p_lifo->p_top,
                                                       (void **)&p_cur_top,
                                                       &cur_tag);
        if (p_cur_top == NULL)
            return NULL;
        ABTI_sync_lifo_element *p_next = p_cur_top->p_next;
        /* tag is incremented to avoid the ABA problem. */
        if (ABTU_likely(ABTD_atomic_bool_cas_weak_tagged_ptr(&p_lifo->p_top,
                                                             p_cur_top, cur_tag,
                                                             p_next,
                                                             cur_tag + 1))) {
            return p_cur_top;
        }
    }
#else
    ABTI_sync_lifo_element *p_ret;
    ABTI_spinlock_acquire(&p_lifo->lock);
    p_ret = ABTI_sync_lifo_pop_unsafe(p_lifo);
    ABTI_spinlock_release(&p_lifo->lock);
    return p_ret;
#endif
}

#endif /* ABTI_SYNC_LIFO_H_INCLUDED */
