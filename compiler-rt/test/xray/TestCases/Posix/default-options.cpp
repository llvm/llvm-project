// RUN: rm -fr %t && mkdir %t && cd %t
// RUN: %clang_xray %s -o a.out
// RUN: %run %t/a.out 2>&1 | FileCheck %s

// REQUIRES: built-in-llvm-tree

extern "C" __attribute__((xray_never_instrument)) const char *
__xray_default_options() {
  return "patch_premain=true:verbosity=1:xray_mode=xray-basic";
}

__attribute__((xray_always_instrument)) void always() {}

int main() { always(); }

// CHECK: =={{[0-9].*}}==XRay: Log file in '{{.*}}'
