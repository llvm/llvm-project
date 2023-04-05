// UNSUPPORTED: target={{.*-windows-gnu}}

// RUN: %clang_cl_asan -Od %p/dll_host.cpp -Fe%t
// RUN: %clang_cl_asan -LD -Od %s -Fe%t.dll
// RUN: %env_asan_opts=detect_stack_use_after_return=1 not %run %t %t.dll 2>&1 | FileCheck %s
// RUN: %clang_cl_asan -LD -Od %s -Fe%t.dll -fsanitize-address-use-after-return=always
// RUN: not %run %t %t.dll 2>&1 | FileCheck %s

#include <malloc.h>

char *x;

void foo() {
  char stack_buffer[42];
  x = &stack_buffer[13];
}

extern "C" __declspec(dllexport)
int test_function() {
  foo();
  *x = 42;
// CHECK: AddressSanitizer: stack-use-after-return
// CHECK: WRITE of size 1 at [[ADDR:.*]] thread T0
// CHECK-NEXT:  test_function{{.*}}dll_stack_use_after_return.cpp:[[@LINE-3]]
// CHECK-NEXT:  main
//
// CHECK: Address [[ADDR]] is located in stack of thread T0 at offset [[OFFSET:.*]] in frame
// CHECK-NEXT: #0 {{.*}} foo{{.*}}dll_stack_use_after_return.cpp
// CHECK: 'stack_buffer'{{.*}} <== Memory access at offset [[OFFSET]] is inside this variable
  return 0;
}

