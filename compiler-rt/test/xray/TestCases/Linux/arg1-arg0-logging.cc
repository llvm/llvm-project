// Allow having both the no-arg and arg1 logging implementation live together,
// and be called in the correct cases.
//
// RUN: rm arg0-arg1-logging-* || true
// RUN: %clangxx_xray -std=c++11 %s -o %t
// RUN: XRAY_OPTIONS="patch_premain=true verbosity=1 xray_logfile_base=arg0-arg1-logging-" %run %t
//
// TODO: Support these in ARM and PPC
// XFAIL: arm || aarch64 || mips
// UNSUPPORTED: powerpc64le

#include "xray/xray_interface.h"
#include <cassert>
#include <cstdio>

using namespace std;

bool arg0loggercalled = false;
void arg0logger(int32_t, XRayEntryType) { arg0loggercalled = true; }

[[clang::xray_always_instrument]] void arg0fn() { printf("hello, arg0!\n"); }

bool arg1loggercalled = false;
void arg1logger(int32_t, XRayEntryType, uint64_t) { arg1loggercalled = true; }

[[ clang::xray_always_instrument, clang::xray_log_args(1) ]] void
arg1fn(uint64_t arg1) {
  printf("hello, arg1!\n");
}

int main(int argc, char *argv[]) {
  __xray_set_handler(arg0logger);
  __xray_set_handler_arg1(arg1logger);
  arg0fn();
  arg1fn(0xcafef00d);
  __xray_remove_handler_arg1();
  __xray_remove_handler();
  assert(arg0loggercalled && arg1loggercalled);
}
