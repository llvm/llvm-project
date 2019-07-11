// RUN: %clang_cl_asan -Od %p/dll_host.cc -Fe%t
// RUN: %clang_cl_asan -Wno-fortify-source -LD -Od %s -Fe%t.dll
// RUN: not %run %t %t.dll 2>&1 | FileCheck %s

// Test that it works correctly even with ICF enabled.
// RUN: %clang_cl_asan -Wno-fortify-source -LD -Od %s -Fe%t.dll -link /OPT:REF /OPT:ICF
// RUN: not %run %t %t.dll 2>&1 | FileCheck %s

#include <stdio.h>
#include <string.h>

extern "C" __declspec(dllexport)
int test_function() {
  char buff[5] = "aaaa";

  memset(buff, 'b', 5);
  if (buff[2] != 'b')
    return 2;
  printf("Initial test OK\n");
  fflush(0);
// CHECK: Initial test OK

  memset(buff, 'c', 6);
// CHECK: AddressSanitizer: stack-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 6 at [[ADDR]] thread T0
// CHECK-NEXT:  __asan_memset
// CHECK-NEXT:  test_function {{.*}}dll_intercept_memset.cc:[[@LINE-4]]
// CHECK: Address [[ADDR]] is located in stack of thread T0 at offset {{.*}} in frame
// CHECK-NEXT:  test_function {{.*}}dll_intercept_memset.cc
// CHECK: 'buff'{{.*}} <== Memory access at offset {{.*}} overflows this variable
  return 0;
}
