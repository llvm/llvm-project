// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %env_asan_opts=detect_leaks=0:allocator_may_return_null=1 %run %t 2>&1 | FileCheck %s
// RUN: %env_asan_opts=detect_leaks=0:allocator_may_return_null=1:malloc_context_size=0 %run %t 2>&1 | FileCheck %s
// REQUIRES: linux, (x86_64-target-arch || aarch64-target-arch)
// UNSUPPORTED: android

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sanitizer/asan_interface.h>

extern "C" {
void *__alloc_token_calloc(size_t, size_t, size_t);
void *__alloc_token_realloc(void *, size_t, size_t);
void *__alloc_token_aligned_alloc(size_t, size_t, size_t);
}

int main() {
  void *p = __alloc_token_calloc(1, 1, 0);
  assert(*static_cast<char *>(p) == 0);
  __asan_describe_address(p);
  void *q = __alloc_token_realloc(p, 64, SIZE_MAX);
  __asan_describe_address(p);
  // CHECK: Allocation token: 0x0
  // CHECK: Allocation token: 0x0
  // CHECK: freed by thread

  memset(q, 1, 64);
  assert(!__alloc_token_realloc(q, SIZE_MAX, 1));
  __asan_describe_address(q);
  free(q);
  __asan_describe_address(q);
  // CHECK: Allocation token: 0xffffffffffffffff
  // CHECK: Allocation token: 0xffffffffffffffff
  // CHECK: freed by thread

  p = __alloc_token_aligned_alloc(4096, 4 << 20, 0x123);
  assert(reinterpret_cast<uintptr_t>(p) % 4096 == 0);
  memset(p, 1, 4 << 20);
  free(p);
  __asan_describe_address(p);
  // CHECK: Allocation token: 0x123

  fprintf(stderr, "ordinary allocation\n");
  p = malloc(64);
  __asan_describe_address(p);
  free(p);
  // CHECK: ordinary allocation
  // CHECK-NOT: Allocation token:
}
