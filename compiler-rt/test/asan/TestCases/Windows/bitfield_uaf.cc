// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <windows.h>

typedef struct _S {
  unsigned int bf1:1;
  unsigned int bf2:2;
  unsigned int bf3:3;
  unsigned int bf4:4;
} S;

void make_access(S *s) {
  s->bf2 = 2;
// CHECK: AddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
// CHECK: READ of size {{[124]}} at [[ADDR]]
// CHECK:   {{#0 .* make_access.*bitfield_uaf.cc}}:[[@LINE-3]]
// CHECK:   {{#1 .* main}}
}

int main(void) {
  S *s = (S*)malloc(sizeof(S));
  free(s);
// CHECK: [[ADDR]] is located 0 bytes inside of 4-byte region
// CHECK-LABEL: freed by thread T0 here:
// CHECK:   {{#0 .* free }}
// CHECK:   {{#1 .* main .*bitfield_uaf.cc}}:[[@LINE-4]]
// CHECK-LABEL: previously allocated by thread T0 here:
// CHECK:   {{#0 .* malloc }}
// CHECK:   {{#1 .* main .*bitfield_uaf.cc}}:[[@LINE-8]]
  make_access(s);
  return 0;
}

