// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <string.h>

int main() {
  char str[] = "Hello";
  if (5 != strlen(str))
    return 1;

  printf("Initial test OK\n");
  fflush(0);
// CHECK: Initial test OK

  str[5] = '!';  // Losing '\0' at the end.
  int len = strlen(str);
// CHECK: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// FIXME: Should be READ of size 1, see issue 155.
// CHECK: READ of size {{[0-9]+}} at [[ADDR]] thread T0
// CHECK:      strlen 
// CHECK-NEXT: main {{.*}}intercept_strlen.cc:[[@LINE-5]]
// CHECK: Address [[ADDR]] is located in stack of thread T0 at offset {{.*}} in frame
// CHECK-NEXT: main {{.*}}intercept_strlen.cc
// CHECK: 'str'{{.*}} <== Memory access at offset {{.*}} overflows this variable
  return len < 6;
}
