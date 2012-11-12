// Check that using the preamble option actually skips the preamble.

// RUN: %clang_cc1 -emit-pch -o %t %S/Inputs/preamble.h -DFOO=f
// RUN: %clang_cc1 -include-pch %t -preamble-bytes=317,1 -DFOO=f -verify %s -emit-llvm -o - | FileCheck %s

float f(int); // Not an error, because we skip this via the preamble!












int g(int x) {
  return FOO(x);
}

// CHECK: call {{.*}} @f(
