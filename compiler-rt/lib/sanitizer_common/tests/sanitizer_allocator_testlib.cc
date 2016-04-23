//===-- sanitizer_allocator_testlib.cc ------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
// Malloc replacement library based on CombinedAllocator.
// The primary purpose of this file is an end-to-end integration test
// for CombinedAllocator.
//===----------------------------------------------------------------------===//
/* Usage:
clang++ -std=c++11 -fno-exceptions  -g -fPIC -I. -I../include -Isanitizer \
 sanitizer_common/tests/sanitizer_allocator_testlib.cc \
 $(\ls sanitizer_common/sanitizer_*.cc | grep -v sanitizer_common_nolibc.cc) \
 -shared -lpthread -o testmalloc.so
LD_PRELOAD=`pwd`/testmalloc.so /your/app
*/
#include "sanitizer_common/sanitizer_allocator.h"
#include "sanitizer_common/sanitizer_common.h"
#include <stddef.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>

#ifndef SANITIZER_MALLOC_HOOK
# define SANITIZER_MALLOC_HOOK(p, s)
#endif

#ifndef SANITIZER_FREE_HOOK
# define SANITIZER_FREE_HOOK(p)
#endif

namespace {
static const uptr kAllocatorSpace = 0x600000000000ULL;
static const uptr kAllocatorSize  =  0x10000000000ULL;  // 1T.

// typedef SizeClassAllocator64<kAllocatorSpace, kAllocatorSize, 0,
typedef SizeClassAllocator64<~(uptr)0, kAllocatorSize, 0,
  CompactSizeClassMap> PrimaryAllocator;
typedef SizeClassAllocatorLocalCache<PrimaryAllocator> AllocatorCache;
typedef LargeMmapAllocator<> SecondaryAllocator;
typedef CombinedAllocator<PrimaryAllocator, AllocatorCache,
          SecondaryAllocator> Allocator;

static Allocator allocator;
static bool global_inited;
static THREADLOCAL AllocatorCache cache;
static THREADLOCAL bool thread_inited;
static pthread_key_t pkey;

static void thread_dtor(void *v) {
  if ((uptr)v != 3) {
    pthread_setspecific(pkey, (void*)((uptr)v + 1));
    return;
  }
  allocator.SwallowCache(&cache);
}

static void NOINLINE thread_init() {
  if (!global_inited) {
    global_inited = true;
    allocator.Init(false /*may_return_null*/);
    pthread_key_create(&pkey, thread_dtor);
  }
  thread_inited = true;
  pthread_setspecific(pkey, (void*)1);
  cache.Init(nullptr);
}
}  // namespace

extern "C" {
void *malloc(size_t size) {
  if (UNLIKELY(!thread_inited))
    thread_init();
  void *p = allocator.Allocate(&cache, size, 8);
  SANITIZER_MALLOC_HOOK(p, size);
  return p;
}

void free(void *p) {
  if (UNLIKELY(!thread_inited))
    thread_init();
  SANITIZER_FREE_HOOK(p);
  allocator.Deallocate(&cache, p);
}

void *calloc(size_t nmemb, size_t size) {
  if (UNLIKELY(!thread_inited))
    thread_init();
  size *= nmemb;
  void *p = allocator.Allocate(&cache, size, 8, false);
  memset(p, 0, size);
  SANITIZER_MALLOC_HOOK(p, size);
  return p;
}

void *realloc(void *p, size_t size) {
  if (UNLIKELY(!thread_inited))
    thread_init();
  if (p) {
    SANITIZER_FREE_HOOK(p);
  }
  p = allocator.Reallocate(&cache, p, size, 8);
  if (p) {
    SANITIZER_MALLOC_HOOK(p, size);
  }
  return p;
}

void *memalign(size_t alignment, size_t size) {
  if (UNLIKELY(!thread_inited))
    thread_init();
  void *p = allocator.Allocate(&cache, size, alignment);
  SANITIZER_MALLOC_HOOK(p, size);
  return p;
}

int posix_memalign(void **memptr, size_t alignment, size_t size) {
  if (UNLIKELY(!thread_inited))
    thread_init();
  *memptr = allocator.Allocate(&cache, size, alignment);
  SANITIZER_MALLOC_HOOK(*memptr, size);
  return 0;
}

void *valloc(size_t size) {
  if (UNLIKELY(!thread_inited))
    thread_init();
  if (size == 0)
    size = GetPageSizeCached();
  void *p = allocator.Allocate(&cache, size, GetPageSizeCached());
  SANITIZER_MALLOC_HOOK(p, size);
  return p;
}

void cfree(void *p) ALIAS("free");
void *pvalloc(size_t size) ALIAS("valloc");
void *__libc_memalign(size_t alignment, size_t size) ALIAS("memalign");

void malloc_usable_size() {
}

void mallinfo() {
}

void mallopt() {
}
}  // extern "C"

namespace std {
  struct nothrow_t;
}

void *operator new(size_t size) ALIAS("malloc");
void *operator new[](size_t size) ALIAS("malloc");
void *operator new(size_t size, std::nothrow_t const&) ALIAS("malloc");
void *operator new[](size_t size, std::nothrow_t const&) ALIAS("malloc");
void operator delete(void *ptr) throw() ALIAS("free");
void operator delete[](void *ptr) throw() ALIAS("free");
void operator delete(void *ptr, std::nothrow_t const&) ALIAS("free");
void operator delete[](void *ptr, std::nothrow_t const&) ALIAS("free");
