// RUN: %clang_cl_asan -O0 %p/dll_host.cc -Fe%t
// RUN: %clang_cl_asan -LD -O0 %s -Fe%t.dll
// RUN: not %run %t %t.dll 2>&1 | FileCheck %s

#include <process.h>

void noreturn_f() {
  int subscript = -1;
  char buffer[42];
  buffer[subscript] = 42;
  _exit(1);
// CHECK: AddressSanitizer: stack-buffer-underflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] thread T0
// CHECK-NEXT:  noreturn_f{{.*}}dll_noreturn.cc:[[@LINE-4]]
// CHECK-NEXT:  test_function{{.*}}dll_noreturn.cc
// CHECK-NEXT:  main{{.*}}dll_host.cc
//
// CHECK: Address [[ADDR]] is located in stack of thread T0 at offset [[OFFSET:.*]] in frame
// CHECK-NEXT:  noreturn_f{{.*}}dll_noreturn.cc
// CHECK: 'buffer'{{.*}} <== Memory access at offset [[OFFSET]] underflows this variable
// CHECK-LABEL: SUMMARY
}

extern "C" __declspec(dllexport)
int test_function() {
  noreturn_f();
  return 0;
}
