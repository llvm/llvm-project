// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

// CHECK: Initial test OK
// CHECK: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 6 at [[ADDR]] thread T0
// CHECK-NEXT:  mem{{.*}}
// CHECK-NEXT:  call_mem{{.*}}
// CHECK-NEXT:  main {{.*}}intercept_memmove.cpp:[[@LINE-5]]
// CHECK: Address [[ADDR]] is located in stack of thread T0 at offset {{.*}} in frame
// CHECK-NEXT:   #0 {{.*}} main
// CHECK: 'buff2'{{.*}} <== Memory access at offset {{.*}} overflows this variable

// The CRT provides memmove in several shapes, depending on CRT version and
// flavor: aliased to memcpy as a single implementation (x64 static CRT),
// a distinct function (x86, and vcruntime140.dll 14.38+ where memcpy became
// a jmp-to-memmove thunk), or resolved from the CRT DLL exports. Calling
// through a function pointer defeats the compiler builtin and verifies that
// the CRT's memmove entry point is intercepted in all these shapes.

#include <stdio.h>
#include <string.h>

void call_memmove(void *(*f)(void *, const void *, size_t), void *a,
                  const void *b, size_t c) {
  f(a, b, c);
}

int main() {
  char buff1[6] = "Hello", buff2[5];
  call_memmove(&memmove, buff2, buff1, 5);
  if (buff1[2] != buff2[2])
    return 2;
  printf("Initial test OK\n");
  fflush(0);
  call_memmove(&memmove, buff2, buff1, 6);
}
