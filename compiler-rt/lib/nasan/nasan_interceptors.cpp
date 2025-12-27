//===-- nasan_interceptors.cpp - NoAliasSanitizer Interceptors -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Interceptors for libc functions and allocators.
//
//===----------------------------------------------------------------------===//

#include "nasan.h"
#include "nasan_internal.h"
#include "sanitizer_common/sanitizer_common.h"
#include "sanitizer_common/sanitizer_libc.h"

#include <dlfcn.h>

// Placement new declaration (can't include <new> due to -nostdinc++)
inline void *operator new(__SIZE_TYPE__, void *__p) noexcept { return __p; }

using namespace __nasan;
using namespace __sanitizer;

// Function pointers to real implementations
typedef void *(*malloc_fn_t)(uptr);
typedef void (*free_fn_t)(void *);
typedef void *(*calloc_fn_t)(uptr, uptr);
typedef void *(*realloc_fn_t)(void *, uptr);
typedef void *(*reallocarray_fn_t)(void *, uptr, uptr);
typedef void *(*aligned_alloc_fn_t)(uptr, uptr);
typedef int (*posix_memalign_fn_t)(void **, uptr, uptr);
typedef void *(*memalign_fn_t)(uptr, uptr);
typedef void *(*valloc_fn_t)(uptr);
typedef char *(*strdup_fn_t)(const char *);
typedef char *(*strndup_fn_t)(const char *, uptr);
typedef void *(*memcpy_fn_t)(void *, const void *, uptr);
typedef void *(*memmove_fn_t)(void *, const void *, uptr);
typedef void *(*memset_fn_t)(void *, int, uptr);

static malloc_fn_t real_malloc = nullptr;
static free_fn_t real_free = nullptr;
static calloc_fn_t real_calloc = nullptr;
static realloc_fn_t real_realloc = nullptr;
static reallocarray_fn_t real_reallocarray = nullptr;
static aligned_alloc_fn_t real_aligned_alloc = nullptr;
static posix_memalign_fn_t real_posix_memalign = nullptr;
static memalign_fn_t real_memalign = nullptr;
static valloc_fn_t real_valloc = nullptr;
static strdup_fn_t real_strdup = nullptr;
static strndup_fn_t real_strndup = nullptr;
static memcpy_fn_t real_memcpy = nullptr;
static memmove_fn_t real_memmove = nullptr;
static memset_fn_t real_memset = nullptr;

static thread_local bool g_in_interceptor = false;

struct InterceptorGuard {
  InterceptorGuard() {
    was_in = g_in_interceptor;
    g_in_interceptor = true;
  }
  ~InterceptorGuard() {
    g_in_interceptor = was_in;
  }
  bool was_in;
};

static void init_interceptors() {
  static volatile bool initialized = false;
  static volatile bool initializing = false;

  if (initialized) return;

  // Prevent recursion and race conditions during initialization
  if (g_in_interceptor) return;
  if (__sync_lock_test_and_set(&initializing, true)) {
    // Another thread is initializing, spin until done
    while (!initialized) {}
    return;
  }

  InterceptorGuard guard;

  real_malloc = (malloc_fn_t)dlsym(RTLD_NEXT, "malloc");
  real_free = (free_fn_t)dlsym(RTLD_NEXT, "free");
  real_calloc = (calloc_fn_t)dlsym(RTLD_NEXT, "calloc");
  real_realloc = (realloc_fn_t)dlsym(RTLD_NEXT, "realloc");
  real_reallocarray = (reallocarray_fn_t)dlsym(RTLD_NEXT, "reallocarray");
  real_aligned_alloc = (aligned_alloc_fn_t)dlsym(RTLD_NEXT, "aligned_alloc");
  real_posix_memalign = (posix_memalign_fn_t)dlsym(RTLD_NEXT, "posix_memalign");
  real_memalign = (memalign_fn_t)dlsym(RTLD_NEXT, "memalign");
  real_valloc = (valloc_fn_t)dlsym(RTLD_NEXT, "valloc");
  real_strdup = (strdup_fn_t)dlsym(RTLD_NEXT, "strdup");
  real_strndup = (strndup_fn_t)dlsym(RTLD_NEXT, "strndup");
  real_memcpy = (memcpy_fn_t)dlsym(RTLD_NEXT, "memcpy");
  real_memmove = (memmove_fn_t)dlsym(RTLD_NEXT, "memmove");
  real_memset = (memset_fn_t)dlsym(RTLD_NEXT, "memset");

  __sync_synchronize();
  initialized = true;
}

// Allocators - create new provenance roots
extern "C" {

void *malloc(uptr size) {
  init_interceptors();

  if (g_in_interceptor) {
    return real_malloc(size);
  }

  InterceptorGuard guard;
  void *ptr = real_malloc(size);

  if (ptr && size > 0) {
    __nasan_create_allocation_provenance(ptr, size);
  }

  return ptr;
}

void free(void *ptr) {
  init_interceptors();

  if (g_in_interceptor || !ptr) {
    if (real_free) real_free(ptr);
    return;
  }

  InterceptorGuard guard;
  __nasan_destroy_allocation_provenance(ptr);
  real_free(ptr);
}

void *calloc(uptr nmemb, uptr size) {
  init_interceptors();

  if (g_in_interceptor) {
    return real_calloc(nmemb, size);
  }

  InterceptorGuard guard;
  void *ptr = real_calloc(nmemb, size);

  if (ptr && nmemb * size > 0) {
    __nasan_create_allocation_provenance(ptr, nmemb * size);
  }

  return ptr;
}

void *realloc(void *old_ptr, uptr size) {
  init_interceptors();

  if (g_in_interceptor) {
    return real_realloc(old_ptr, size);
  }

  InterceptorGuard guard;

  // Destroy old provenance
  if (old_ptr) {
    __nasan_destroy_allocation_provenance(old_ptr);
  }

  void *ptr = real_realloc(old_ptr, size);

  // Create new provenance
  if (ptr && size > 0) {
    __nasan_create_allocation_provenance(ptr, size);
  }

  return ptr;
}

#if !SANITIZER_APPLE
void *reallocarray(void *old_ptr, uptr nmemb, uptr size) {
  init_interceptors();

  if (g_in_interceptor || !real_reallocarray) {
    if (real_reallocarray)
      return real_reallocarray(old_ptr, nmemb, size);
    // Fallback to realloc
    return realloc(old_ptr, nmemb * size);
  }

  InterceptorGuard guard;

  if (old_ptr) {
    __nasan_destroy_allocation_provenance(old_ptr);
  }

  void *ptr = real_reallocarray(old_ptr, nmemb, size);

  if (ptr && nmemb * size > 0) {
    __nasan_create_allocation_provenance(ptr, nmemb * size);
  }

  return ptr;
}
#endif

void *aligned_alloc(uptr alignment, uptr size) {
  init_interceptors();

  if (g_in_interceptor || !real_aligned_alloc) {
    if (real_aligned_alloc)
      return real_aligned_alloc(alignment, size);
    return nullptr;
  }

  InterceptorGuard guard;
  void *ptr = real_aligned_alloc(alignment, size);

  if (ptr && size > 0) {
    __nasan_create_allocation_provenance(ptr, size);
  }

  return ptr;
}

int posix_memalign(void **memptr, uptr alignment, uptr size) {
  init_interceptors();

  if (g_in_interceptor || !real_posix_memalign) {
    if (real_posix_memalign)
      return real_posix_memalign(memptr, alignment, size);
    return -1;
  }

  InterceptorGuard guard;
  int result = real_posix_memalign(memptr, alignment, size);

  if (result == 0 && *memptr && size > 0) {
    __nasan_create_allocation_provenance(*memptr, size);
  }

  return result;
}

#if !SANITIZER_APPLE
void *memalign(uptr alignment, uptr size) {
  init_interceptors();

  if (g_in_interceptor || !real_memalign) {
    if (real_memalign)
      return real_memalign(alignment, size);
    return nullptr;
  }

  InterceptorGuard guard;
  void *ptr = real_memalign(alignment, size);

  if (ptr && size > 0) {
    __nasan_create_allocation_provenance(ptr, size);
  }

  return ptr;
}

void *valloc(uptr size) {
  init_interceptors();

  if (g_in_interceptor || !real_valloc) {
    if (real_valloc)
      return real_valloc(size);
    return nullptr;
  }

  InterceptorGuard guard;
  void *ptr = real_valloc(size);

  if (ptr && size > 0) {
    __nasan_create_allocation_provenance(ptr, size);
  }

  return ptr;
}
#endif

char *strdup(const char *s) {
  init_interceptors();

  if (g_in_interceptor || !real_strdup) {
    if (real_strdup)
      return real_strdup(s);
    return nullptr;
  }

  InterceptorGuard guard;
  char *ptr = real_strdup(s);

  if (ptr) {
    uptr len = internal_strlen(s) + 1;
    __nasan_create_allocation_provenance(ptr, len);
  }

  return ptr;
}

char *strndup(const char *s, uptr n) {
  init_interceptors();

  if (g_in_interceptor || !real_strndup) {
    if (real_strndup)
      return real_strndup(s, n);
    return nullptr;
  }

  InterceptorGuard guard;
  char *ptr = real_strndup(s, n);

  if (ptr) {
    uptr len = internal_strlen(ptr) + 1;
    __nasan_create_allocation_provenance(ptr, len);
  }

  return ptr;
}

// Memory operations - propagate provenance and check accesses
void *memcpy(void *dest, const void *src, uptr n) {
  init_interceptors();

  if (g_in_interceptor || n == 0) {
    return real_memcpy(dest, src, n);
  }

  InterceptorGuard guard;

  // Check access using both dest and src pointers
  ProvenanceID dest_prov = __nasan_get_pointer_provenance(dest);
  ProvenanceID src_prov = __nasan_get_pointer_provenance(const_cast<void *>(src));
  __nasan_check_store(reinterpret_cast<u64>(dest), n, dest_prov);
  __nasan_check_load(reinterpret_cast<u64>(src), n, src_prov);

  // Perform actual copy
  void *result = real_memcpy(dest, src, n);

  // Propagate provenance for any pointers copied (conservative approach)
  NASanThreadState *state = get_thread_state();
  for (uptr i = 0; i < n; i += sizeof(void *)) {
    if (i + sizeof(void *) <= n) {
      void *src_addr = static_cast<char *>(const_cast<void *>(src)) + i;
      void *dst_addr = static_cast<char *>(dest) + i;

      auto *entry = state->stored_pointer_provenance.find(src_addr);
      if (entry) {
        state->stored_pointer_provenance[dst_addr] = entry->second;
      }
    }
  }

  return result;
}

void *memmove(void *dest, const void *src, uptr n) {
  init_interceptors();

  if (g_in_interceptor || n == 0) {
    return real_memmove(dest, src, n);
  }

  InterceptorGuard guard;

  ProvenanceID dest_prov = __nasan_get_pointer_provenance(dest);
  ProvenanceID src_prov = __nasan_get_pointer_provenance(const_cast<void *>(src));
  __nasan_check_store(reinterpret_cast<u64>(dest), n, dest_prov);
  __nasan_check_load(reinterpret_cast<u64>(src), n, src_prov);

  void *result = real_memmove(dest, src, n);

  // Propagate provenance (same as memcpy)
  NASanThreadState *state = get_thread_state();
  for (uptr i = 0; i < n; i += sizeof(void *)) {
    if (i + sizeof(void *) <= n) {
      void *src_addr = static_cast<char *>(const_cast<void *>(src)) + i;
      void *dst_addr = static_cast<char *>(dest) + i;

      auto *entry = state->stored_pointer_provenance.find(src_addr);
      if (entry) {
        state->stored_pointer_provenance[dst_addr] = entry->second;
      }
    }
  }

  return result;
}

void *memset(void *s, int c, uptr n) {
  init_interceptors();

  if (g_in_interceptor || n == 0) {
    return real_memset(s, c, n);
  }

  InterceptorGuard guard;

  ProvenanceID prov = __nasan_get_pointer_provenance(s);
  __nasan_check_store(reinterpret_cast<u64>(s), n, prov);

  void *result = real_memset(s, c, n);

  // memset clears memory - clear stored provenance
  NASanThreadState *state = get_thread_state();
  for (uptr i = 0; i < n; i += sizeof(void *)) {
    void *addr = static_cast<char *>(s) + i;
    state->stored_pointer_provenance[addr] = 0;  // Clear provenance
  }

  return result;
}

} // extern "C"

// C++ operators new/delete
void *operator new(uptr size) {
  void *ptr = malloc(size);
  if (!ptr) {
    Printf("NASan: operator new failed to allocate %zu bytes\n", size);
    Die();
  }
  return ptr;
}

void *operator new[](uptr size) {
  void *ptr = malloc(size);
  if (!ptr) {
    Printf("NASan: operator new[] failed to allocate %zu bytes\n", size);
    Die();
  }
  return ptr;
}

void operator delete(void *ptr) noexcept {
  free(ptr);
}

void operator delete[](void *ptr) noexcept {
  free(ptr);
}

void operator delete(void *ptr, uptr) noexcept {
  free(ptr);
}

void operator delete[](void *ptr, uptr) noexcept {
  free(ptr);
}
