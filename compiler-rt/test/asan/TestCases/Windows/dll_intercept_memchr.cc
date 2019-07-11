// RUN: %clang_cl_asan -Od %p/dll_host.cc -Fe%t
// RUN: %clang_cl_asan -LD -Od %s -Fe%t.dll
// RUN: not %run %t %t.dll 2>&1 | FileCheck %s

// On windows 64-bit, the memchr function is written in assembly and is not
// hookable with the interception library. There is not enough padding before
// the function and there is a short jump on the second instruction which
// doesn't not allow enough space to encode a 64-bit indirect jump.
// UNSUPPORTED: x86_64-windows

#include <string.h>

extern "C" __declspec(dllexport)
int test_function() {
  char buff[6] = "Hello";

  memchr(buff, 'z', 7);
// CHECK: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: READ of size 7 at [[ADDR]] thread T0
// CHECK-NEXT:  __asan_wrap_memchr
// CHECK-NEXT:  memchr
// CHECK-NEXT:  test_function {{.*}}dll_intercept_memchr.cc:[[@LINE-5]]
// CHECK: Address [[ADDR]] is located in stack of thread T0 at offset {{.*}} in frame
// CHECK-NEXT:  test_function {{.*}}dll_intercept_memchr.cc
// CHECK: 'buff'{{.*}} <== Memory access at offset {{.*}} overflows this variable
  return 0;
}
