// RUN: %clang_xray -g -fxray-modes=xray-basic,xray-fdr,xray-profiling -o %t %s
// RUN: rm -f xray-log.c-test.*
// RUN: XRAY_OPTIONS=patch_premain=true:verbosity=1:xray_mode=xray-basic %t \
// RUN:     2>&1 | FileCheck %s
// RUN: rm -f xray-log.c-test.*
//
// REQUIRES: target={{(aarch64|loongarch64|x86_64)-.*}}
// REQUIRES: built-in-llvm-tree
__attribute__((xray_always_instrument)) void always() {}

int main() {
  always();
}

// CHECK: =={{[0-9].*}}==XRay: Log file in '{{.*}}'
