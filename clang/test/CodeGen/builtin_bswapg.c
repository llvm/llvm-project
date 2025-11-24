// RUN: %clang_cc1 -triple x86_64 -O0 -emit-llvm -o - %s | FileCheck %s
#include <stdbool.h>
bool test_bswapg(bool c) {
  return __builtin_bswapg(c);
}

// CHECK-LABEL: define{{.*}} i1 @_Z11test_bswapgb(
// CHECK: ret i1 %{{.*}}
