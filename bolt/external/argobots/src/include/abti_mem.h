/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTI_MEM_H_INCLUDED
#define ABTI_MEM_H_INCLUDED

/* Memory allocation */

/* Round desc_size up to the cacheline size.  The last four bytes will be
 * used to determine whether the descriptor is allocated externally (i.e.,
 * malloc()) or taken from a memory pool. */
#define ABTI_MEM_POOL_DESC_ELEM_SIZE                                           \
    ((sizeof(ABTI_thread) + ABT_CONFIG_STATIC_CACHELINE_SIZE - 1) &            \
     (~(ABT_CONFIG_STATIC_CACHELINE_SIZE - 1)))

enum {
    ABTI_MEM_LP_MALLOC = 0,
    ABTI_MEM_LP_MMAP_RP,
    ABTI_MEM_LP_MMAP_HP_RP,
    ABTI_MEM_LP_MMAP_HP_THP,
    ABTI_MEM_LP_THP
};

void ABTI_mem_init(ABTI_global *p_global);
void ABTI_mem_init_local(ABTI_xstream *p_local_xstream);
void ABTI_mem_finalize(ABTI_global *p_global);
void ABTI_mem_finalize_local(ABTI_xstream *p_local_xstream);
int ABTI_mem_check_lp_alloc(int lp_alloc);

/* Inline functions */
ABTU_ret_err static inline int
ABTI_mem_alloc_nythread_malloc(ABTI_thread **pp_thread)
{
    ABTI_thread *p_thread;
    int abt_errno =
        ABTU_malloc(ABTI_MEM_POOL_DESC_ELEM_SIZE, (void **)&p_thread);
    ABTI_CHECK_ERROR(abt_errno);
    p_thread->type = ABTI_THREAD_TYPE_MEM_MALLOC_DESC;
    *pp_thread = p_thread;
    return ABT_SUCCESS;
}

#ifdef ABT_CONFIG_USE_MEM_POOL
ABTU_ret_err static inline int
ABTI_mem_alloc_nythread_mempool(ABTI_local *p_local, ABTI_thread **pp_thread)
{
    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
    if (ABTI_IS_EXT_THREAD_ENABLED && p_local_xstream == NULL) {
        /* For external threads */
        return ABTI_mem_alloc_nythread_malloc(pp_thread);
    }
    /* Find the page that has an empty block */
    ABTI_thread *p_thread;
    int abt_errno = ABTI_mem_pool_alloc(&p_local_xstream->mem_pool_desc,
                                        (void **)&p_thread);
    ABTI_CHECK_ERROR(abt_errno);
    p_thread->type = ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC;
    *pp_thread = p_thread;
    return ABT_SUCCESS;
}
#endif

ABTU_ret_err static inline int ABTI_mem_alloc_nythread(ABTI_local *p_local,
                                                       ABTI_thread **pp_thread)
{
#ifdef ABT_CONFIG_USE_MEM_POOL
    return ABTI_mem_alloc_nythread_mempool(p_local, pp_thread);
#else
    return ABTI_mem_alloc_nythread_malloc(pp_thread);
#endif
}

static inline void ABTI_mem_free_nythread(ABTI_local *p_local,
                                          ABTI_thread *p_thread)
{
    /* Return stack. */
#ifdef ABT_CONFIG_USE_MEM_POOL
    if (p_thread->type & ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC) {
        ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
        /* Came from a memory pool. */
#ifndef ABT_CONFIG_DISABLE_EXT_THREAD
        if (p_local_xstream == NULL) {
            /* Return a stack to the global pool. */
            ABTI_spinlock_acquire(&gp_ABTI_global->mem_pool_desc_lock);
            ABTI_mem_pool_free(&gp_ABTI_global->mem_pool_desc_ext, p_thread);
            ABTI_spinlock_release(&gp_ABTI_global->mem_pool_desc_lock);
            return;
        }
#endif
        ABTI_mem_pool_free(&p_local_xstream->mem_pool_desc, p_thread);
        return;
    }
#endif
    /* p_thread was allocated by malloc() */
    ABTU_free(p_thread);
}

#ifdef ABT_CONFIG_USE_MEM_POOL
ABTU_ret_err static inline int ABTI_mem_alloc_ythread_mempool_desc_stack_impl(
    ABTI_mem_pool_local_pool *p_mem_pool_stack, size_t stacksize,
    ABTI_ythread **pp_ythread, void **pp_stack)
{
    /* stacksize must be a multiple of ABT_CONFIG_STATIC_CACHELINE_SIZE. */
    ABTI_ASSERT((stacksize & (ABT_CONFIG_STATIC_CACHELINE_SIZE - 1)) == 0);
    void *p_ythread;
    int abt_errno = ABTI_mem_pool_alloc(p_mem_pool_stack, &p_ythread);
    ABTI_CHECK_ERROR(abt_errno);

    *pp_stack = (void *)(((char *)p_ythread) - stacksize);
    *pp_ythread = (ABTI_ythread *)p_ythread;
    return ABT_SUCCESS;
}
#endif

ABTU_ret_err static inline int ABTI_mem_alloc_ythread_malloc_desc_stack_impl(
    size_t stacksize, ABTI_ythread **pp_ythread, void **pp_stack)
{
    /* stacksize must be a multiple of ABT_CONFIG_STATIC_CACHELINE_SIZE. */
    size_t alloc_stacksize =
        (stacksize + ABT_CONFIG_STATIC_CACHELINE_SIZE - 1) &
        (~(ABT_CONFIG_STATIC_CACHELINE_SIZE - 1));
    char *p_stack;
    int abt_errno =
        ABTU_malloc(alloc_stacksize + sizeof(ABTI_ythread), (void **)&p_stack);
    ABTI_CHECK_ERROR(abt_errno);

    *pp_stack = (void *)p_stack;
    *pp_ythread = (ABTI_ythread *)(p_stack + alloc_stacksize);
    return ABT_SUCCESS;
}

ABTU_ret_err static inline int
ABTI_mem_alloc_ythread_default(ABTI_local *p_local, ABTI_ythread **pp_ythread)
{
    size_t stacksize = gp_ABTI_global->thread_stacksize;
    ABTI_ythread *p_ythread;
    void *p_stack;
    /* If an external thread allocates a stack, we use ABTU_malloc. */
    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
    if (ABTI_IS_EXT_THREAD_ENABLED && p_local_xstream == NULL) {
        int abt_errno =
            ABTI_mem_alloc_ythread_malloc_desc_stack_impl(stacksize, &p_ythread,
                                                          &p_stack);
        ABTI_CHECK_ERROR(abt_errno);
        p_ythread->thread.type = ABTI_THREAD_TYPE_MEM_MALLOC_DESC_STACK;
    } else {
#ifdef ABT_CONFIG_USE_MEM_POOL
        int abt_errno = ABTI_mem_alloc_ythread_mempool_desc_stack_impl(
            &p_local_xstream->mem_pool_stack, stacksize, &p_ythread, &p_stack);
        ABTI_CHECK_ERROR(abt_errno);
        p_ythread->thread.type = ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC_STACK;
#else
        int abt_errno =
            ABTI_mem_alloc_ythread_malloc_desc_stack_impl(stacksize, &p_ythread,
                                                          &p_stack);
        ABTI_CHECK_ERROR(abt_errno);
        p_ythread->thread.type = ABTI_THREAD_TYPE_MEM_MALLOC_DESC_STACK;
#endif
    }
    /* Initialize members of ABTI_thread_attr. */
    p_ythread->p_stack = p_stack;
    p_ythread->stacksize = stacksize;
    ABTI_VALGRIND_REGISTER_STACK(p_ythread->p_stack, p_ythread->stacksize);
    *pp_ythread = p_ythread;
    return ABT_SUCCESS;
}

#ifdef ABT_CONFIG_USE_MEM_POOL
ABTU_ret_err static inline int ABTI_mem_alloc_ythread_mempool_desc_stack(
    ABTI_local *p_local, ABTI_thread_attr *p_attr, ABTI_ythread **pp_ythread)
{
    size_t stacksize = gp_ABTI_global->thread_stacksize;
    ABTI_ythread *p_ythread;
    void *p_stack;
    /* If an external thread allocates a stack, we use ABTU_malloc. */
    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
    if (ABTI_IS_EXT_THREAD_ENABLED && p_local_xstream == NULL) {
        int abt_errno =
            ABTI_mem_alloc_ythread_malloc_desc_stack_impl(stacksize, &p_ythread,
                                                          &p_stack);
        ABTI_CHECK_ERROR(abt_errno);
        p_ythread->thread.type = ABTI_THREAD_TYPE_MEM_MALLOC_DESC_STACK;
    } else {
        int abt_errno = ABTI_mem_alloc_ythread_mempool_desc_stack_impl(
            &p_local_xstream->mem_pool_stack, stacksize, &p_ythread, &p_stack);
        ABTI_CHECK_ERROR(abt_errno);
        p_ythread->thread.type = ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC_STACK;
    }
    /* Copy members of p_attr. */
    p_ythread->p_stack = p_stack;
    p_ythread->stacksize = stacksize;
    ABTI_VALGRIND_REGISTER_STACK(p_ythread->p_stack, p_ythread->stacksize);
    *pp_ythread = p_ythread;
    return ABT_SUCCESS;
}
#endif

ABTU_ret_err static inline int
ABTI_mem_alloc_ythread_malloc_desc_stack(ABTI_thread_attr *p_attr,
                                         ABTI_ythread **pp_ythread)
{
    size_t stacksize = p_attr->stacksize;
    ABTI_ythread *p_ythread;
    void *p_stack;
    int abt_errno =
        ABTI_mem_alloc_ythread_malloc_desc_stack_impl(stacksize, &p_ythread,
                                                      &p_stack);
    ABTI_CHECK_ERROR(abt_errno);

    /* Copy members of p_attr. */
    p_ythread->thread.type = ABTI_THREAD_TYPE_MEM_MALLOC_DESC_STACK;
    p_ythread->stacksize = stacksize;
    p_ythread->p_stack = p_stack;
    ABTI_VALGRIND_REGISTER_STACK(p_ythread->p_stack, p_ythread->stacksize);
    *pp_ythread = p_ythread;
    return ABT_SUCCESS;
}

ABTU_ret_err static inline int ABTI_mem_alloc_ythread_mempool_desc(
    ABTI_local *p_local, ABTI_thread_attr *p_attr, ABTI_ythread **pp_ythread)
{
    ABTI_ythread *p_ythread;
    if (sizeof(ABTI_ythread) <= ABTI_MEM_POOL_DESC_ELEM_SIZE) {
        /* Use a descriptor pool for ABT_thread. */
        ABTI_STATIC_ASSERT(offsetof(ABTI_ythread, thread) == 0);
        int abt_errno =
            ABTI_mem_alloc_nythread(p_local, (ABTI_thread **)&p_ythread);
        ABTI_CHECK_ERROR(abt_errno);
    } else {
        /* Do not allocate stack, but Valgrind registration is preferred. */
        int abt_errno = ABTU_malloc(sizeof(ABTI_ythread), (void **)&p_ythread);
        ABTI_CHECK_ERROR(abt_errno);
        p_ythread->thread.type = ABTI_THREAD_TYPE_MEM_MALLOC_DESC;
    }
    /* Copy members of p_attr. */
    p_ythread->stacksize = p_attr->stacksize;
    p_ythread->p_stack = p_attr->p_stack;
    /* Note that the valgrind registration is ignored iff p_stack is NULL. */
    ABTI_VALGRIND_REGISTER_STACK(p_ythread->p_stack, p_ythread->stacksize);
    *pp_ythread = p_ythread;
    return ABT_SUCCESS;
}

static inline void ABTI_mem_free_thread(ABTI_local *p_local,
                                        ABTI_thread *p_thread)
{
    /* Return stack. */
#ifdef ABT_CONFIG_USE_MEM_POOL
    if (p_thread->type & ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC_STACK) {
        ABTI_ythread *p_ythread = ABTI_thread_get_ythread(p_thread);
        ABTI_VALGRIND_UNREGISTER_STACK(p_ythread->p_stack);

        ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
        /* Came from a memory pool. */
#ifndef ABT_CONFIG_DISABLE_EXT_THREAD
        if (p_local_xstream == NULL) {
            /* Return a stack to the global pool. */
            ABTI_spinlock_acquire(&gp_ABTI_global->mem_pool_stack_lock);
            ABTI_mem_pool_free(&gp_ABTI_global->mem_pool_stack_ext, p_ythread);
            ABTI_spinlock_release(&gp_ABTI_global->mem_pool_stack_lock);
            return;
        }
#endif
        ABTI_mem_pool_free(&p_local_xstream->mem_pool_stack, p_ythread);
    } else
#endif
        if (p_thread->type & ABTI_THREAD_TYPE_MEM_MEMPOOL_DESC) {
        /* Non-yieldable thread or yieldable thread without stack. */
#ifdef HAVE_VALGRIND_SUPPORT
        ABTI_ythread *p_ythread = ABTI_thread_get_ythread_or_null(p_thread);
        if (p_ythread)
            ABTI_VALGRIND_UNREGISTER_STACK(p_ythread->p_stack);
#endif
        ABTI_mem_free_nythread(p_local, p_thread);
    } else if (p_thread->type & ABTI_THREAD_TYPE_MEM_MALLOC_DESC_STACK) {
        ABTI_ythread *p_ythread = ABTI_thread_get_ythread(p_thread);
        ABTI_VALGRIND_UNREGISTER_STACK(p_ythread->p_stack);
        ABTU_free(p_ythread->p_stack);
    } else {
        ABTI_ASSERT(p_thread->type & ABTI_THREAD_TYPE_MEM_MALLOC_DESC);
        ABTI_STATIC_ASSERT(offsetof(ABTI_ythread, thread) == 0);
#ifdef HAVE_VALGRIND_SUPPORT
        ABTI_ythread *p_ythread = ABTI_thread_get_ythread_or_null(p_thread);
        if (p_ythread)
            ABTI_VALGRIND_UNREGISTER_STACK(p_ythread->p_stack);
#endif
        ABTU_free(p_thread);
    }
}

/* Generic scalable memory pools.  It uses a memory pool for ABTI_thread.
 * The last four bytes will be used to determine whether the descriptor is
 * allocated externally (i.e., malloc()) or taken from a memory pool. */
#define ABTI_MEM_POOL_DESC_SIZE (ABTI_MEM_POOL_DESC_ELEM_SIZE - 4)

ABTU_ret_err static inline int ABTI_mem_alloc_desc(ABTI_local *p_local,
                                                   void **pp_desc)
{
#ifndef ABT_CONFIG_USE_MEM_POOL
    return ABTU_malloc(ABTI_MEM_POOL_DESC_SIZE, pp_desc);
#else
    void *p_desc;
    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
    if (ABTI_IS_EXT_THREAD_ENABLED && p_local_xstream == NULL) {
        /* For external threads */
        int abt_errno = ABTU_malloc(ABTI_MEM_POOL_DESC_SIZE, &p_desc);
        ABTI_CHECK_ERROR(abt_errno);
        *(uint32_t *)(((char *)p_desc) + ABTI_MEM_POOL_DESC_SIZE) = 1;
        *pp_desc = p_desc;
        return ABT_SUCCESS;
    } else {
        /* Find the page that has an empty block */
        int abt_errno =
            ABTI_mem_pool_alloc(&p_local_xstream->mem_pool_desc, &p_desc);
        ABTI_CHECK_ERROR(abt_errno);
        /* To distinguish it from a malloc'ed case, assign non-NULL value. */
        *(uint32_t *)(((char *)p_desc) + ABTI_MEM_POOL_DESC_SIZE) = 0;
        *pp_desc = p_desc;
        return ABT_SUCCESS;
    }
#endif
}

static inline void ABTI_mem_free_desc(ABTI_local *p_local, void *p_desc)
{
#ifndef ABT_CONFIG_USE_MEM_POOL
    ABTU_free(p_desc);
#else
    ABTI_xstream *p_local_xstream = ABTI_local_get_xstream_or_null(p_local);
#ifndef ABT_CONFIG_DISABLE_EXT_THREAD
    if (*(uint32_t *)(((char *)p_desc) + ABTI_MEM_POOL_DESC_SIZE)) {
        /* This was allocated by an external thread. */
        ABTU_free(p_desc);
        return;
    } else if (!p_local_xstream) {
        /* Return a stack and a descriptor to their global pools. */
        ABTI_spinlock_acquire(&gp_ABTI_global->mem_pool_desc_lock);
        ABTI_mem_pool_free(&gp_ABTI_global->mem_pool_desc_ext, p_desc);
        ABTI_spinlock_release(&gp_ABTI_global->mem_pool_desc_lock);
        return;
    }
#endif
    ABTI_mem_pool_free(&p_local_xstream->mem_pool_desc, p_desc);
#endif
}

#endif /* ABTI_MEM_H_INCLUDED */
