/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#include <sys/types.h>
#include <sys/mman.h>

#include "abti.h"

#define ABTU_LP_PROTS (PROT_READ | PROT_WRITE)

#if defined(HAVE_MAP_ANONYMOUS)
#define ABTU_LP_FLAGS_RP (MAP_PRIVATE | MAP_ANONYMOUS)
#define ABTU_LP_USE_MMAP 1
#elif defined(HAVE_MAP_ANON)
#define ABTU_LP_FLAGS_RP (MAP_PRIVATE | MAP_ANON)
#define ABTU_LP_USE_MMAP 1
#else
#define ABTU_LP_USE_MMAP 0
#endif

#if ABTU_LP_USE_MMAP && defined(HAVE_MAP_HUGETLB)
#define ABTU_LP_FLAGS_HP (ABTU_LP_FLAGS_RP | MAP_HUGETLB)
#define ABTU_LP_USE_MMAP_HUGEPAGE 1
#else
/* NOTE: On Mac OS, we tried VM_FLAGS_SUPERPAGE_SIZE_ANY that is defined in
 * <mach/vm_statistics.h>, but mmap() failed with it and its execution was too
 * slow.  By that reason, we do not support it for now. */
#define ABTU_LP_USE_HUGEPAGE 0
#endif

static void *mmap_regular(size_t size)
{
#if ABTU_LP_USE_MMAP
    void *p_page = mmap(NULL, size, ABTU_LP_PROTS, ABTU_LP_FLAGS_RP, 0, 0);
    return p_page != MAP_FAILED ? p_page : NULL;
#else
    return NULL;
#endif
}

static void *mmap_hugepage(size_t size)
{
#if ABTU_LP_USE_MMAP && ABTU_LP_USE_MMAP_HUGEPAGE
    void *p_page = mmap(NULL, size, ABTU_LP_PROTS, ABTU_LP_FLAGS_HP, 0, 0);
    return p_page != MAP_FAILED ? p_page : NULL;
#else
    return NULL;
#endif
}

static void mmap_free(void *p_page, size_t size)
{
    munmap(p_page, size);
}

/* Returns if a given large page type is supported. */
int ABTU_is_supported_largepage_type(size_t size, size_t alignment_hint,
                                     ABTU_MEM_LARGEPAGE_TYPE requested)
{
    if (requested == ABTU_MEM_LARGEPAGE_MALLOC) {
        /* It always succeeds. */
        return 1;
    } else if (requested == ABTU_MEM_LARGEPAGE_MEMALIGN) {
        void *p_page;
        int ret = ABTU_memalign(alignment_hint, size, &p_page);
        if (ret == ABT_SUCCESS && p_page != NULL) {
            ABTU_free(p_page);
            return 1;
        }
    } else if (requested == ABTU_MEM_LARGEPAGE_MMAP) {
        void *p_page = mmap_regular(size);
        if (p_page) {
            mmap_free(p_page, size);
            return 1;
        }
    } else if (requested == ABTU_MEM_LARGEPAGE_MMAP_HUGEPAGE) {
        void *p_page = mmap_hugepage(size);
        if (p_page) {
            mmap_free(p_page, size);
            return 1;
        }
    }
    /* Not supported. */
    return 0;
}

ABTU_ret_err int
ABTU_alloc_largepage(size_t size, size_t alignment_hint,
                     const ABTU_MEM_LARGEPAGE_TYPE *requested_types,
                     int num_requested_types, ABTU_MEM_LARGEPAGE_TYPE *p_actual,
                     void **p_ptr)
{
    int i;
    void *ptr = NULL;
    for (i = 0; i < num_requested_types; i++) {
        ABTU_MEM_LARGEPAGE_TYPE requested = requested_types[i];
        if (requested == ABTU_MEM_LARGEPAGE_MALLOC) {
            int abt_error = ABTU_malloc(size, &ptr);
            if (abt_error == ABT_SUCCESS && ptr) {
                *p_actual = ABTU_MEM_LARGEPAGE_MALLOC;
                *p_ptr = ptr;
                return ABT_SUCCESS;
            }
        } else if (requested == ABTU_MEM_LARGEPAGE_MEMALIGN) {
            int abt_error = ABTU_memalign(alignment_hint, size, &ptr);
            if (abt_error == ABT_SUCCESS && ptr) {
                *p_actual = ABTU_MEM_LARGEPAGE_MEMALIGN;
                *p_ptr = ptr;
                return ABT_SUCCESS;
            }
        } else if (requested == ABTU_MEM_LARGEPAGE_MMAP) {
            ptr = mmap_regular(size);
            if (ptr) {
                *p_actual = ABTU_MEM_LARGEPAGE_MMAP;
                *p_ptr = ptr;
                return ABT_SUCCESS;
            }
        } else if (requested == ABTU_MEM_LARGEPAGE_MMAP_HUGEPAGE) {
            ptr = mmap_hugepage(size);
            if (ptr) {
                *p_actual = ABTU_MEM_LARGEPAGE_MMAP_HUGEPAGE;
                *p_ptr = ptr;
                return ABT_SUCCESS;
            }
        }
    }
    return ABT_ERR_MEM;
}

void ABTU_free_largepage(void *ptr, size_t size, ABTU_MEM_LARGEPAGE_TYPE type)
{
    if (!ptr)
        return;
    if (type == ABTU_MEM_LARGEPAGE_MALLOC) {
        ABTU_free(ptr);
    } else if (type == ABTU_MEM_LARGEPAGE_MEMALIGN) {
        ABTU_free(ptr);
    } else if (type == ABTU_MEM_LARGEPAGE_MMAP) {
        mmap_free(ptr, size);
    } else if (type == ABTU_MEM_LARGEPAGE_MMAP_HUGEPAGE) {
        mmap_free(ptr, size);
    }
}
