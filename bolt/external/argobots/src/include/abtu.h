/* -*- Mode: C; c-basic-offset:4 ; indent-tabs-mode:nil ; -*- */
/*
 * See COPYRIGHT in top-level directory.
 */

#ifndef ABTU_H_INCLUDED
#define ABTU_H_INCLUDED

#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "abt_config.h"

/* Utility feature */

#ifdef HAVE___BUILTIN_EXPECT
#define ABTU_likely(cond) __builtin_expect(!!(cond), 1)
#define ABTU_unlikely(cond) __builtin_expect(!!(cond), 0)
#else
#define ABTU_likely(cond) (cond)
#define ABTU_unlikely(cond) (cond)
#endif

#ifdef HAVE___BUILTIN_UNREACHABLE
#define ABTU_unreachable() __builtin_unreachable()
#else
#define ABTU_unreachable()
#endif

#ifdef HAVE_FUNC_ATTRIBUTE_NORETURN
#define ABTU_noreturn __attribute__((noreturn))
#else
#define ABTU_noreturn
#endif

#ifdef ABT_CONFIG_HAVE_ALIGNOF_GCC
#define ABTU_alignof(type) (__alignof__(type))
#elif defined(ABT_CONFIG_HAVE_ALIGNOF_C11)
#define ABTU_alignof(type) (alignof(type))
#else
#define ABTU_alignof(type) 16 /* 16 bytes would be a good guess. */
#endif
#define ABTU_MAX_ALIGNMENT                                                     \
    (ABTU_alignof(long double) > ABTU_alignof(long long)                       \
         ? ABTU_alignof(long double)                                           \
         : ABTU_alignof(long long))

#ifdef HAVE_FUNC_ATTRIBUTE_WARN_UNUSED_RESULT
#define ABTU_ret_err __attribute__((warn_unused_result))
#else
#define ABTU_ret_err
#endif

/*
 * An attribute to hint an alignment of a member variable.
 * Usage:
 * struct X {
 *   void *obj_1;
 *   ABTU_align_member_var(64)
 *   void *obj_2;
 * };
 */
#ifndef __SUNPRO_C
#define ABTU_align_member_var(size) __attribute__((aligned(size)))
#else
/* Sun Studio does not support it. */
#define ABTU_align_member_var(size)
#endif

/*
 * An attribute to suppress address sanitizer warning.
 */
#if defined(__GNUC__) && defined(__SANITIZE_ADDRESS__)
/*
 * Older GCC cannot combine no_sanitize_address + always_inline (e.g., builtin
 * memcpy on some platforms), which causes *a compilation error*.  Let's accept
 * false-positive warning if used GCC is old.  This issue seems fixed between
 * GCC 7.4.0 and GCC 8.3.0 as far as I checked.
 * https://gcc.gnu.org/bugzilla/show_bug.cgi?id=59600
 */
#if __GNUC__ >= 8
#define ABTU_no_sanitize_address __attribute__((no_sanitize_address))
#endif
#elif __clang__
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#if __clang_major__ >= 4 || (__clang_major__ >= 3 && __clang_minor__ >= 7)
/* >= Clang 3.7.0 */
#define ABTU_no_sanitize_address __attribute__((no_sanitize("address")))
#elif (__clang_major__ >= 3 && __clang_minor__ >= 3)
/* >= Clang 3.3.0 */
#define ABTU_no_sanitize_address __attribute__((no_sanitize_address))
#elif (__clang_major__ >= 3 && __clang_minor__ >= 1)
/* >= Clang 3.1.0 */
#define ABTU_no_sanitize_address __attribute__((no_address_safety_analysis))
#else /* Too old clang. */
#define ABTU_no_sanitize_address
#endif
#endif /* __has_feature(address_sanitizer) */
#endif /* defined(__has_feature) */
#endif

#ifndef ABTU_no_sanitize_address
/* We do not support other address sanitizers. */
#define ABTU_no_sanitize_address
#endif

/* Utility Functions */

ABTU_ret_err static inline int ABTU_memalign(size_t alignment, size_t size,
                                             void **p_ptr)
{
    void *ptr;
    int ret = posix_memalign(&ptr, alignment, size);
    if (ABTI_IS_ERROR_CHECK_ENABLED && ret != 0) {
        return ABT_ERR_MEM;
    }
    *p_ptr = ptr;
    return ABT_SUCCESS;
}

static inline void ABTU_free(void *ptr)
{
    free(ptr);
}

#ifdef ABT_CONFIG_USE_ALIGNED_ALLOC

ABTU_ret_err static inline int ABTU_malloc(size_t size, void **p_ptr)
{
    /* Round up to the smallest multiple of ABT_CONFIG_STATIC_CACHELINE_SIZE
     * which is greater than or equal to size in order to avoid any
     * false-sharing. */
    size = (size + ABT_CONFIG_STATIC_CACHELINE_SIZE - 1) &
           (~(ABT_CONFIG_STATIC_CACHELINE_SIZE - 1));
    return ABTU_memalign(ABT_CONFIG_STATIC_CACHELINE_SIZE, size, p_ptr);
}

ABTU_ret_err static inline int ABTU_calloc(size_t num, size_t size,
                                           void **p_ptr)
{
    void *ptr;
    int ret = ABTU_malloc(num * size, &ptr);
    if (ABTI_IS_ERROR_CHECK_ENABLED && ret != ABT_SUCCESS) {
        return ABT_ERR_MEM;
    }
    memset(ptr, 0, num * size);
    *p_ptr = ptr;
    return ABT_SUCCESS;
}

ABTU_ret_err static inline int ABTU_realloc(size_t old_size, size_t new_size,
                                            void **p_ptr)
{
    void *new_ptr, *old_ptr = *p_ptr;
    int ret = ABTU_malloc(new_size, &new_ptr);
    if (ABTI_IS_ERROR_CHECK_ENABLED && ret != ABT_SUCCESS) {
        return ABT_ERR_MEM;
    }
    memcpy(new_ptr, old_ptr, (old_size < new_size) ? old_size : new_size);
    ABTU_free(old_ptr);
    *p_ptr = new_ptr;
    return ABT_SUCCESS;
}

#else /* ABT_CONFIG_USE_ALIGNED_ALLOC */

ABTU_ret_err static inline int ABTU_malloc(size_t size, void **p_ptr)
{
    void *ptr = malloc(size);
    if (ABTI_IS_ERROR_CHECK_ENABLED && ptr == NULL) {
        return ABT_ERR_MEM;
    }
    *p_ptr = ptr;
    return ABT_SUCCESS;
}

ABTU_ret_err static inline int ABTU_calloc(size_t num, size_t size,
                                           void **p_ptr)
{
    void *ptr = calloc(num, size);
    if (ABTI_IS_ERROR_CHECK_ENABLED && ptr == NULL) {
        return ABT_ERR_MEM;
    }
    *p_ptr = ptr;
    return ABT_SUCCESS;
}

ABTU_ret_err static inline int ABTU_realloc(size_t old_size, size_t new_size,
                                            void **p_ptr)
{
    (void)old_size;
    void *ptr = realloc(*p_ptr, new_size);
    if (ABTI_IS_ERROR_CHECK_ENABLED && ptr == NULL) {
        return ABT_ERR_MEM;
    }
    *p_ptr = ptr;
    return ABT_SUCCESS;
}

#endif /* !ABT_CONFIG_USE_ALIGNED_ALLOC */

typedef enum ABTU_MEM_LARGEPAGE_TYPE {
    ABTU_MEM_LARGEPAGE_MALLOC,   /* ABTU_malloc(). */
    ABTU_MEM_LARGEPAGE_MEMALIGN, /* memalign() */
    ABTU_MEM_LARGEPAGE_MMAP,     /* normal private memory obtained by mmap() */
    ABTU_MEM_LARGEPAGE_MMAP_HUGEPAGE, /* hugepage obtained by mmap() */
} ABTU_MEM_LARGEPAGE_TYPE;

/* Returns 1 if a given large page type is supported. */
int ABTU_is_supported_largepage_type(size_t size, size_t alignment_hint,
                                     ABTU_MEM_LARGEPAGE_TYPE requested);
ABTU_ret_err int
ABTU_alloc_largepage(size_t size, size_t alignment_hint,
                     const ABTU_MEM_LARGEPAGE_TYPE *requested_types,
                     int num_requested_types, ABTU_MEM_LARGEPAGE_TYPE *p_actual,
                     void **p_ptr);
void ABTU_free_largepage(void *ptr, size_t size, ABTU_MEM_LARGEPAGE_TYPE type);

#endif /* ABTU_H_INCLUDED */
