// RUN: %clang_cl_asan -Od %p/dll_host.cpp -Fe%t
// RUN: %clang_cl_asan -LD -Od %s -Fe%t.dll
// RUN: not %run %t %t.dll 2>&1 | FileCheck %s

#include <malloc.h>
extern "C" __declspec(dllexport)
int test_function() {
  char *buffer = (char*)malloc(42);
  buffer[-1] = 42;
// CHECK: AddressSanitizer: heap-buffer-overflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] thread T0
// CHECK-NEXT: test_function {{.*}}dll_malloc_left_oob.cpp:[[@LINE-3]]
// CHECK-NEXT: main {{.*}}dll_host.cpp
//
// CHECK: [[ADDR]] is located 1 bytes to the left of 42-byte region
// CHECK-LABEL: allocated by thread T0 here:
// CHECK-NEXT:   malloc
// CHECK-NEXT:   test_function {{.*}}dll_malloc_left_oob.cpp:[[@LINE-10]]
// CHECK-NEXT:   main {{.*}}dll_host.cpp
// CHECK-LABEL: SUMMARY
  free(buffer);
  return 0;
}
