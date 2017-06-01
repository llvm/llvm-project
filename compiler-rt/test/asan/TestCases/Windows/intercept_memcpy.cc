// RUN: %clang_cl_asan -O0 %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <stdio.h>
#include <string.h>

void call_memcpy(void* (*f)(void *, const void *, size_t),
                 void *a, const void *b, size_t c) {
  f(a, b, c);
}

int main() {
  char buff1[6] = "Hello", buff2[5];

  call_memcpy(&memcpy, buff2, buff1, 5);
  if (buff1[2] != buff2[2])
    return 2;
  printf("Initial test OK\n");
  fflush(0);
// CHECK: Initial test OK

  call_memcpy(&memcpy, buff2, buff1, 6);
// CHECK: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 6 at [[ADDR]] thread T0
// CHECK-NEXT:  __asan_{{.*}}mem{{.*}}
// CHECK-NEXT:  call_mem{{.*}}
// CHECK-NEXT:  main {{.*}}intercept_memcpy.cc:[[@LINE-5]]
// CHECK: Address [[ADDR]] is located in stack of thread T0 at offset {{.*}} in frame
// CHECK-NEXT:   #0 {{.*}} main
// CHECK: 'buff2'{{.*}} <== Memory access at offset {{.*}} overflows this variable
}
