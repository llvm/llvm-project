// RUN: %clang_cl_asan -Od %p/dll_host.cpp -Fe%t
// RUN: %clang_cl_asan -LD -Od %s -Fe%t.dll
// RUN: not %run %t %t.dll 2>&1 | FileCheck %s

#include <sanitizer/asan_interface.h>

void should_not_crash(volatile char *c) {
  *c = 42;
}

void should_crash(volatile char *c) {
  *c = 42;
}

extern "C" __declspec(dllexport)
int test_function() {
  char buffer[256];
  should_not_crash(&buffer[0]);
  __asan_poison_memory_region(buffer, 128);
  should_not_crash(&buffer[192]);
  __asan_unpoison_memory_region(buffer, 64);
  should_not_crash(&buffer[32]);

  should_crash(&buffer[96]);
// CHECK: AddressSanitizer: use-after-poison on address [[ADDR:0x[0-9a-f]+]]
// CHECK-NEXT: WRITE of size 1 at [[ADDR]] thread T0
// CHECK-NEXT: should_crash{{.*}}\dll_poison_unpoison.cpp
// CHECK-NEXT: test_function{{.*}}\dll_poison_unpoison.cpp:[[@LINE-4]]
// CHECK-NEXT: main
//
// CHECK: [[ADDR]] is located in stack of thread T0 at offset [[OFFSET:.*]] in frame
// CHECK-NEXT: test_function{{.*}}\dll_poison_unpoison.cpp
// CHECK: 'buffer'{{.*}} <== Memory access at offset [[OFFSET]] is inside this variable
  return 0;
}
