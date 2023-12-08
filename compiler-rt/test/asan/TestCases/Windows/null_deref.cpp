// RUN: %clangxx_asan -O0 %s -o %t && not %run %t 2>&1 | FileCheck %s
// FIXME: merge this with the common null_deref test when we can run common
// tests on Windows.

__attribute__((noinline))
static void NullDeref(int *ptr) {
  // CHECK: ERROR: AddressSanitizer: access-violation on unknown address
  // CHECK:   {{0x0*000.. .*pc 0x.*}}
  ptr[10]++;  // BOOM
}
int main() {
  NullDeref((int*)0);
  // CHECK: {{    #1 0x.* in main.*null_deref.cpp:}}[[@LINE-1]]:3
  // CHECK: AddressSanitizer can not provide additional info.
}
