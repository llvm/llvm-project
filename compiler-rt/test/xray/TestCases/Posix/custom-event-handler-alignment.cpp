// Make sure we're aligning the stack properly when lowering the custom event
// calls.
//
// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: XRAY_OPTIONS="patch_premain=false verbosity=1" \
// RUN:     %run %t 2>&1
// REQUIRES: x86_64-target-arch
// REQUIRES: built-in-llvm-tree
#include <xmmintrin.h>
#include <stdio.h>
#include "xray/xray_interface.h"

[[clang::xray_never_instrument]] __attribute__((weak)) __m128 f(__m128 *i) {
  return *i;
}

[[clang::xray_always_instrument]] void foo() {
  __xray_customevent(0, 0);
  __m128 v = {};
  f(&v);
}

[[clang::xray_always_instrument]] void bar() {
  __xray_customevent(0, 0);
}

void printer(void* ptr, size_t size) {
  printf("handler called\n");
  __m128 v = {};
  f(&v);
}

int main(int argc, char* argv[]) {
  __xray_set_customevent_handler(printer);
  __xray_patch();
  foo();  // CHECK: handler called
  bar();  // CHECK: handler called
  __xray_unpatch();
  __xray_remove_customevent_handler();
  foo();
  bar();
}
