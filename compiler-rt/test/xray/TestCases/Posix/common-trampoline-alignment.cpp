// Make sure that we're aligning the stack properly to support handlers that
// expect 16-byte alignment of the stack.
//
// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: XRAY_OPTIONS="patch_premain=false verbosity=1" \
// RUN:     %run %t 2>&1
// REQUIRES: x86_64-target-arch
// REQUIRES: built-in-llvm-tree
#include "xray/xray_interface.h"
#include <stdio.h>
#include <xmmintrin.h>

[[clang::xray_never_instrument]] __attribute__((weak)) __m128 f(__m128 *i) {
  return *i;
}

[[clang::xray_always_instrument]] __attribute__((noinline)) void noarg() {
  __m128 v = {};
  f(&v);
}

[[ clang::xray_always_instrument, clang::xray_log_args(1) ]]
__attribute__((noinline)) void arg1(int) {
  __m128 v = {};
  f(&v);
}

[[clang::xray_always_instrument]] __attribute__((noinline))
void no_alignment() {}

[[clang::xray_never_instrument]] void noarg_handler(int32_t,
                                                        XRayEntryType) {
  printf("noarg handler called\n");
  __m128 v = {};
  f(&v);
}

[[clang::xray_never_instrument]] void arg1_handler(int32_t, XRayEntryType,
                                                   uint64_t) {
  printf("arg1 handler called\n");
  __m128 v = {};
  f(&v);
}

int main(int argc, char *argv[]) {
  __xray_set_handler(noarg_handler);
  __xray_set_handler_arg1(arg1_handler);
  __xray_patch();
  noarg();    // CHECK: noarg handler called
  arg1(argc); // CHECK: arg1 handler called
  no_alignment();
  __xray_unpatch();
  __xray_remove_handler();
  __xray_remove_handler_arg1();
  noarg();
  arg1(argc);
}
