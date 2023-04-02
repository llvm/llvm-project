// UNSUPPORTED: target={{.*-windows-gnu}}

// RUN: %clang_cl_asan -Od %p/dll_host.cpp -Fe%t
// RUN: %clang_cl_asan -LD -Od %s -Fe%t.dll
// RUN: not %run %t %t.dll 2>&1 | FileCheck %s

#include <malloc.h>

extern "C" __declspec(dllexport)
int test_function() {
  int *buffer = (int*)malloc(42);
  free(buffer);
  buffer[0] = 42;
// CHECK: AddressSanitizer: heap-use-after-free on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 4 at [[ADDR]] thread T0
// CHECK-NEXT:  test_function {{.*}}dll_malloc_uaf.cpp:[[@LINE-3]]
// CHECK-NEXT:  main {{.*}}dll_host
//
// CHECK: [[ADDR]] is located 0 bytes inside of 42-byte region
// CHECK-LABEL: freed by thread T0 here:
// CHECK-NEXT:  free
// CHECK-NEXT:  test_function {{.*}}dll_malloc_uaf.cpp:[[@LINE-10]]
// CHECK-NEXT:  main {{.*}}dll_host
//
// CHECK-LABEL: previously allocated by thread T0 here:
// CHECK-NEXT:  malloc
// CHECK-NEXT:  test_function {{.*}}dll_malloc_uaf.cpp:[[@LINE-16]]
// CHECK-NEXT:  main {{.*}}dll_host
  return 0;
}
