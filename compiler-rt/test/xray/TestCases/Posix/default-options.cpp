// RUN: %clang_xray -g -fxray-default-options='patch_premain=true:verbosity=1:xray_mode=xray-basic' -o %t %s
// RUN: rm -f xray-log.default-options.*
// RUN: %run %t 2>&1 | FileCheck %s
// RUN: rm -f xray-log.default-options.*
//
// REQUIRES: target={{(aarch64|loongarch64|x86_64)-.*}}
// REQUIRES: built-in-llvm-tree
__attribute__((xray_always_instrument)) void always() {}

int main() { always(); }

// CHECK: =={{[0-9].*}}==XRay: Log file in '{{.*}}'
