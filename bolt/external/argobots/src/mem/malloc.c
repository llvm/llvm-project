/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include "abti.h"

#ifdef ABT_CONFIG_USE_MEM_POOL
/* Currently the total memory allocated for stacks and task block pages is not
 * shrunk to avoid the thrashing overhead except that ESs are terminated or
 * ABT_finalize is called.  When an ES terminates its execution, stacks and
 * empty pages that it holds are deallocated.  Non-empty pages are added to the
 * global data.  When ABTI_finalize is called, all memory objects that we have
 * allocated are returned to the higher-level memory allocator. */

void ABTI_mem_init(ABTI_global *p_global)
{
    int num_requested_types = 0;
    ABTU_MEM_LARGEPAGE_TYPE requested_types[3];
    switch (gp_ABTI_global->mem_lp_alloc) {
        case ABTI_MEM_LP_MMAP_RP:
            requested_types[num_requested_types++] = ABTU_MEM_LARGEPAGE_MMAP;
            requested_types[num_requested_types++] = ABTU_MEM_LARGEPAGE_MALLOC;
            break;
        case ABTI_MEM_LP_MMAP_HP_RP:
            requested_types[num_requested_types++] =
                ABTU_MEM_LARGEPAGE_MMAP_HUGEPAGE;
            requested_types[num_requested_types++] = ABTU_MEM_LARGEPAGE_MMAP;
            requested_types[num_requested_types++] = ABTU_MEM_LARGEPAGE_MALLOC;
            break;
        case ABTI_MEM_LP_MMAP_HP_THP:
            requested_types[num_requested_types++] =
                ABTU_MEM_LARGEPAGE_MMAP_HUGEPAGE;
            requested_types[num_requested_types++] =
                ABTU_MEM_LARGEPAGE_MEMALIGN;
            requested_types[num_requested_types++] = ABTU_MEM_LARGEPAGE_MALLOC;
            break;
        case ABTI_MEM_LP_THP:
            requested_types[num_requested_types++] =
                ABTU_MEM_LARGEPAGE_MEMALIGN;
            requested_types[num_requested_types++] = ABTU_MEM_LARGEPAGE_MALLOC;
            break;
        default:
            requested_types[num_requested_types++] = ABTU_MEM_LARGEPAGE_MALLOC;
            break;
    }
    size_t thread_stacksize = p_global->thread_stacksize;
    ABTI_ASSERT((thread_stacksize & (ABT_CONFIG_STATIC_CACHELINE_SIZE - 1)) ==
                0);
    size_t stacksize = (thread_stacksize + sizeof(ABTI_ythread) +
                        ABT_CONFIG_STATIC_CACHELINE_SIZE - 1) &
                       (~(ABT_CONFIG_STATIC_CACHELINE_SIZE - 1));
    if ((stacksize & (2 * ABT_CONFIG_STATIC_CACHELINE_SIZE - 1)) == 0) {
        /* Avoid a multiple of 2 * cacheline size to avoid cache bank conflict.
         */
        stacksize += ABT_CONFIG_STATIC_CACHELINE_SIZE;
    }
    ABTI_mem_pool_init_global_pool(&p_global->mem_pool_stack,
                                   p_global->mem_max_stacks /
                                       ABT_MEM_POOL_MAX_LOCAL_BUCKETS,
                                   stacksize, thread_stacksize,
                                   p_global->mem_sp_size, requested_types,
                                   num_requested_types,
                                   gp_ABTI_global->mem_page_size);
    /* The last four bytes will be used to store a mempool flag */
    ABTI_STATIC_ASSERT((ABTI_MEM_POOL_DESC_ELEM_SIZE &
                        (ABT_CONFIG_STATIC_CACHELINE_SIZE - 1)) == 0);
    ABTI_mem_pool_init_global_pool(&p_global->mem_pool_desc,
                                   p_global->mem_max_descs /
                                       ABT_MEM_POOL_MAX_LOCAL_BUCKETS,
                                   ABTI_MEM_POOL_DESC_ELEM_SIZE, 0,
                                   p_global->mem_page_size, requested_types,
                                   num_requested_types,
                                   gp_ABTI_global->mem_page_size);
#ifndef ABT_CONFIG_DISABLE_EXT_THREAD
    ABTI_spinlock_clear(&p_global->mem_pool_stack_lock);
    ABTI_mem_pool_init_local_pool(&p_global->mem_pool_stack_ext,
                                  &p_global->mem_pool_stack);
    ABTI_spinlock_clear(&p_global->mem_pool_desc_lock);
    ABTI_mem_pool_init_local_pool(&p_global->mem_pool_desc_ext,
                                  &p_global->mem_pool_desc);
#endif
}

void ABTI_mem_init_local(ABTI_xstream *p_local_xstream)
{
    ABTI_mem_pool_init_local_pool(&p_local_xstream->mem_pool_stack,
                                  &gp_ABTI_global->mem_pool_stack);
    ABTI_mem_pool_init_local_pool(&p_local_xstream->mem_pool_desc,
                                  &gp_ABTI_global->mem_pool_desc);
}

void ABTI_mem_finalize(ABTI_global *p_global)
{
#ifndef ABT_CONFIG_DISABLE_EXT_THREAD
    ABTI_mem_pool_destroy_local_pool(&p_global->mem_pool_stack_ext);
    ABTI_mem_pool_destroy_local_pool(&p_global->mem_pool_desc_ext);
#endif
    ABTI_mem_pool_destroy_global_pool(&p_global->mem_pool_stack);
    ABTI_mem_pool_destroy_global_pool(&p_global->mem_pool_desc);
}

void ABTI_mem_finalize_local(ABTI_xstream *p_local_xstream)
{
    ABTI_mem_pool_destroy_local_pool(&p_local_xstream->mem_pool_stack);
    ABTI_mem_pool_destroy_local_pool(&p_local_xstream->mem_pool_desc);
}

int ABTI_mem_check_lp_alloc(int lp_alloc)
{
    size_t sp_size = gp_ABTI_global->mem_sp_size;
    size_t pg_size = gp_ABTI_global->mem_page_size;
    size_t alignment = ABT_CONFIG_STATIC_CACHELINE_SIZE;
    switch (lp_alloc) {
        case ABTI_MEM_LP_MMAP_RP:
            if (ABTU_is_supported_largepage_type(pg_size, alignment,
                                                 ABTU_MEM_LARGEPAGE_MMAP)) {
                return ABTI_MEM_LP_MMAP_RP;
            } else {
                return ABTI_MEM_LP_MALLOC;
            }
        case ABTI_MEM_LP_MMAP_HP_RP:
            if (ABTU_is_supported_largepage_type(
                    sp_size, alignment, ABTU_MEM_LARGEPAGE_MMAP_HUGEPAGE)) {
                return ABTI_MEM_LP_MMAP_HP_RP;
            } else if (
                ABTU_is_supported_largepage_type(pg_size, alignment,
                                                 ABTU_MEM_LARGEPAGE_MMAP)) {
                return ABTI_MEM_LP_MMAP_RP;
            } else {
                return ABTI_MEM_LP_MALLOC;
            }
        case ABTI_MEM_LP_MMAP_HP_THP:
            if (ABTU_is_supported_largepage_type(
                    sp_size, alignment, ABTU_MEM_LARGEPAGE_MMAP_HUGEPAGE)) {
                return ABTI_MEM_LP_MMAP_HP_THP;
            } else if (
                ABTU_is_supported_largepage_type(pg_size,
                                                 gp_ABTI_global->huge_page_size,
                                                 ABTU_MEM_LARGEPAGE_MEMALIGN)) {
                return ABTI_MEM_LP_THP;
            } else {
                return ABTI_MEM_LP_MALLOC;
            }
        case ABTI_MEM_LP_THP:
            if (ABTU_is_supported_largepage_type(pg_size,
                                                 gp_ABTI_global->huge_page_size,
                                                 ABTU_MEM_LARGEPAGE_MEMALIGN)) {
                return ABTI_MEM_LP_THP;
            } else {
                return ABTI_MEM_LP_MALLOC;
            }
        default:
            return ABTI_MEM_LP_MALLOC;
    }
}

#else /* !ABT_CONFIG_USE_MEM_POOL */

void ABTI_mem_init(ABTI_global *p_global)
{
}

void ABTI_mem_init_local(ABTI_xstream *p_local_xstream)
{
}

void ABTI_mem_finalize(ABTI_global *p_global)
{
}

void ABTI_mem_finalize_local(ABTI_xstream *p_local_xstream)
{
}

#endif /* !ABT_CONFIG_USE_MEM_POOL */
