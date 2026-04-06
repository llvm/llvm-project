// Verify that vec_malloc, vec_calloc, vec_realloc, and the plain allocators
// all return 16-byte aligned memory on AIX.
//
// Note: 16-byte alignment is structurally guaranteed by DefaultSizeClassMap
// (kMinSizeLog=4, so every size class is a multiple of 16) combined with the
// minimum 16-byte redzone.  This test therefore documents/verifies the AIX ABI
// guarantee end-to-end rather than catching regressions in the explicit
// alignment=16 argument passed by asan_vec_malloc/asan_vec_calloc.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: %run %t 2>&1 | FileCheck %s

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static void check_align(const char *label, void *ptr) {
  if (((size_t)ptr & 15) != 0) {
    fprintf(stderr, "FAIL: %s ptr %p not 16-byte aligned\n", label, ptr);
    exit(1);
  }
  fprintf(stderr, "OK: %s\n", label);
}

int main() {
  void *vm = vec_malloc(32);
  check_align("vec_malloc", vm);
  // CHECK: OK: vec_malloc

  void *vc = vec_calloc(1, 32);
  check_align("vec_calloc", vc);
  // CHECK: OK: vec_calloc

  void *vr_null = vec_realloc(NULL, 32);
  check_align("vec_realloc(NULL)", vr_null);
  // CHECK: OK: vec_realloc(NULL)

  void *vr_vm = vec_realloc(vm, 64);
  check_align("vec_realloc(vec_malloc_ptr)", vr_vm);
  // CHECK: OK: vec_realloc(vec_malloc_ptr)

  void *vr_vc = vec_realloc(vc, 64);
  check_align("vec_realloc(vec_calloc_ptr)", vr_vc);
  // CHECK: OK: vec_realloc(vec_calloc_ptr)

  void *m = malloc(32);
  check_align("malloc", m);
  // CHECK: OK: malloc

  void *c = calloc(1, 32);
  check_align("calloc", c);
  // CHECK: OK: calloc

  void *re_null = realloc(NULL, 32);
  check_align("realloc(NULL)", re_null);
  // CHECK: OK: realloc(NULL)

  void *re_m = realloc(m, 64);
  check_align("realloc(malloc_ptr)", re_m);
  // CHECK: OK: realloc(malloc_ptr)

  void *vm2 = vec_malloc(32);
  void *re_vm = realloc(vm2, 64);
  check_align("realloc(vec_malloc_ptr)", re_vm);
  // CHECK: OK: realloc(vec_malloc_ptr)

  free(vr_null);
  free(vr_vm);
  free(vr_vc);
  free(c);
  free(re_null);
  free(re_m);
  free(re_vm);
  return 0;
}
