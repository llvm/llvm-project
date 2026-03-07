// Verify vec_malloc, vec_calloc and vec_realloc interceptors, and that all
// vec/plain allocators return 16-byte aligned memory on AIX.

// RUN: %clangxx_asan -O0 %s -o %t
// RUN: not %run %t vec_malloc  2>&1 | FileCheck %s --check-prefix=CHECK-MALLOC
// RUN: not %run %t vec_calloc  2>&1 | FileCheck %s --check-prefix=CHECK-CALLOC
// RUN: not %run %t vec_realloc 2>&1 | FileCheck %s --check-prefix=CHECK-REALLOC
// RUN: %run %t align           2>&1 | FileCheck %s --check-prefix=CHECK-ALIGN

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Helper used by the "align" sub-test: prints OK/FAIL to stderr.
static void check_align(const char *label, void *ptr) {
  if (((size_t)ptr & 15) != 0) {
    fprintf(stderr, "FAIL: %s ptr %p not 16-byte aligned\n", label, ptr);
    exit(1);
  }
  fprintf(stderr, "OK: %s\n", label);
}

int main(int argc, char **argv) {
  if (argc != 2)
    return 1;

  // --- overflow detection (existing) ---

  char *p;
  if (strcmp(argv[1], "vec_malloc") == 0)
    p = (char *)vec_malloc(10);
  // CHECK-MALLOC: {{READ of size 1 at 0x.* thread T0}}
  // CHECK-MALLOC: {{0x.* is located 0 bytes after 10-byte region}}
  // CHECK-MALLOC: {{0x.* in .vec_malloc}}
  else if (strcmp(argv[1], "vec_calloc") == 0)
    p = (char *)vec_calloc(10, 1);
  // CHECK-CALLOC: {{READ of size 1 at 0x.* thread T0}}
  // CHECK-CALLOC: {{0x.* is located 0 bytes after 10-byte region}}
  // CHECK-CALLOC: {{0x.* in .vec_calloc}}
  else if (strcmp(argv[1], "vec_realloc") == 0)
    p = (char *)vec_realloc(NULL, 10);
  // CHECK-REALLOC: {{READ of size 1 at 0x.* thread T0}}
  // CHECK-REALLOC: {{0x.* is located 0 bytes after 10-byte region}}
  // On AIX 32-bit vec_realloc resolves to the realloc symbol; match both.
  // CHECK-REALLOC: {{0x.* in .*realloc}}

  // --- 16-byte alignment checks ---

  else if (strcmp(argv[1], "align") == 0) {
    void *vm = vec_malloc(32);
    check_align("vec_malloc", vm);
    // CHECK-ALIGN: OK: vec_malloc

    void *vc = vec_calloc(1, 32);
    check_align("vec_calloc", vc);
    // CHECK-ALIGN: OK: vec_calloc

    void *vr_null = vec_realloc(NULL, 32);
    check_align("vec_realloc(NULL)", vr_null);
    // CHECK-ALIGN: OK: vec_realloc(NULL)

    void *vr_vm = vec_realloc(vm, 64);
    check_align("vec_realloc(vec_malloc_ptr)", vr_vm);
    // CHECK-ALIGN: OK: vec_realloc(vec_malloc_ptr)

    void *vr_vc = vec_realloc(vc, 64);
    check_align("vec_realloc(vec_calloc_ptr)", vr_vc);
    // CHECK-ALIGN: OK: vec_realloc(vec_calloc_ptr)

    void *m = malloc(32);
    check_align("malloc", m);
    // CHECK-ALIGN: OK: malloc

    void *c = calloc(1, 32);
    check_align("calloc", c);
    // CHECK-ALIGN: OK: calloc

    void *re_null = realloc(NULL, 32);
    check_align("realloc(NULL)", re_null);
    // CHECK-ALIGN: OK: realloc(NULL)

    void *re_m = realloc(m, 64);
    check_align("realloc(malloc_ptr)", re_m);
    // CHECK-ALIGN: OK: realloc(malloc_ptr)

    void *vm2 = vec_malloc(32);
    void *re_vm = realloc(vm2, 64);
    check_align("realloc(vec_malloc_ptr)", re_vm);
    // CHECK-ALIGN: OK: realloc(vec_malloc_ptr)

    free(vr_null);
    free(vr_vm);
    free(vr_vc);
    free(c);
    free(re_null);
    free(re_m);
    free(re_vm);
    return 0;
  } else {
    return 1;
  }

  char x = p[10];
  free(p);

  return x;
}