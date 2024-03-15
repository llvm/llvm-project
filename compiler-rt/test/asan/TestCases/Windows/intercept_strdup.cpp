// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <string.h>
#include <malloc.h>

int main() {
  char *ptr = _strdup("Hello");
  int subscript = 1;
  ptr[subscript] = '3';
  printf("%s\n", ptr);
  fflush(0);
// CHECK: H3llo

  subscript = -1;
  ptr[subscript] = 42;
  // CHECK: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
  // CHECK: WRITE of size 1 at [[ADDR]] thread T0
  // CHECK:   {{#0 .* main .*}}intercept_strdup.cpp:[[@LINE-3]]
  // CHECK: [[ADDR]] is located 1 bytes before 6-byte region
  // CHECK: allocated by thread T0 here:
  //
  // The first frame is our wrapper normally but will be malloc in the dynamic
  // config.
  // CHECK:   #0 {{.*}} in {{malloc|_strdup}}
  //
  // The local call to _strdup above may be the second or third frame depending
  // on whether we're using the dynamic config.
  // CHECK:   #{{[12]}} {{.*}} in main {{.*}}intercept_strdup.cpp:[[@LINE-21]]
  free(ptr);
}
