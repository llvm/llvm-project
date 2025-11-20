// Intercept the implicit 'this' argument of class member functions.
//
// RUN: %clangxx_xray -g -std=c++11 %s -o %t
// RUN: rm -f log-args-this-*
// RUN: env XRAY_OPTIONS="patch_premain=true verbosity=1 xray_logfile_base=log-args-this-" %run %t

// REQUIRES: target={{(aarch64|x86_64)-.*}}

#include "xray/xray_interface.h"
#include <cassert>

class A {
 public:
  [[clang::xray_always_instrument, clang::xray_log_args(1)]] void f() {
    // does nothing.
  }
};

volatile uint64_t captured = 0;

void handler(int32_t, XRayEntryType, uint64_t arg1) {
  captured = arg1;
}

int main() {
  __xray_set_handler_arg1(handler);
  A instance;
  instance.f();
  __xray_remove_handler_arg1();
  assert(captured == (uint64_t)&instance);
}
