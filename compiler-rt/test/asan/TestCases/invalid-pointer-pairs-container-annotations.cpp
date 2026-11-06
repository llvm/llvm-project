// Pointers into one object must not be reported as an invalid pair just
// because container annotations poison part of the object between them.

// RUN: %clangxx_asan -O0 %s -o %t -mllvm -asan-detect-invalid-pointer-pair

// RUN: %env_asan_opts=detect_invalid_pointer_pairs=2 %run %t ok
// RUN: %env_asan_opts=detect_invalid_pointer_pairs=2:detect_stack_use_after_return=0 %run %t ok
// RUN: %env_asan_opts=detect_invalid_pointer_pairs=2:halt_on_error=0 %run %t bad 2>&1 | FileCheck %s
// XFAIL: msvc

#include <sanitizer/common_interface_defs.h>
#include <stdlib.h>

__attribute__((noinline)) int cmp(char *p, char *q) { return p < q; }
__attribute__((noinline)) long sub(char *p, char *q) { return p - q; }

// Behave like a vector with 4 of 32 bytes in use: [p + 4, p + 32) is poisoned.
static void Annotate(char *p) {
  __sanitizer_annotate_contiguous_container(p, p + 32, p + 32, p + 4);
}
static void Unannotate(char *p) {
  __sanitizer_annotate_contiguous_container(p, p + 32, p + 4, p + 32);
}

static void Check(char *p) {
  cmp(p + 4, p + 32); // size end vs. capacity end
  sub(p + 32, p + 4);
  cmp(p, p + 32);
  sub(p + 32, p);
  cmp(p + 8, p + 16);
  sub(p + 16, p + 8);
}

char global_buf[32];
char global1[32], global2[32];

int main(int argc, char **argv) {
  if (argv[1][0] == 'o') {
    char *heap = (char *)malloc(32);
    Annotate(heap);
    Check(heap);
    Unannotate(heap);
    free(heap);

    char stack_buf[32];
    Annotate(stack_buf);
    Check(stack_buf);
    Unannotate(stack_buf);

    Annotate(global_buf);
    Check(global_buf);
    Unannotate(global_buf);

    // One past the end of a large stack object.
    char stack_large[4096];
    cmp(stack_large, stack_large + 4096);
    sub(stack_large + 4096, stack_large);
    return 0;
  }

  // Annotate one object of each pair so the identity checks are reached.
  char *heap1 = (char *)malloc(32);
  char *heap2 = (char *)malloc(32);
  Annotate(heap1);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-container-annotations.cpp:[[@LINE+1]]
  cmp(heap1, heap2);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-container-annotations.cpp:[[@LINE+1]]
  sub(heap2, heap1 + 4);
  Unannotate(heap1);
  free(heap1);
  free(heap2);

  char stack1[32], stack2[32];
  Annotate(stack1);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-container-annotations.cpp:[[@LINE+1]]
  cmp(stack1, stack2);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-container-annotations.cpp:[[@LINE+1]]
  sub(stack2, stack1 + 4);
  Unannotate(stack1);

  Annotate(global1);

  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-container-annotations.cpp:[[@LINE+1]]
  cmp(global1, global2);
  // CHECK: ERROR: AddressSanitizer: invalid-pointer-pair
  // CHECK: #{{[0-9]+ .*}} in main {{.*}}invalid-pointer-pairs-container-annotations.cpp:[[@LINE+1]]
  sub(global2, global1 + 4);
  Unannotate(global1);
  return 0;
}
