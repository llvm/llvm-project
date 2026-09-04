// RUN: %clang_cc1 -triple arm64-apple-macosx -fblocks -emit-llvm -O0 -o - %s | FileCheck %s

// A block body under an FP-affecting pragma must be emitted in strict-FP mode.

#pragma STDC FENV_ACCESS ON

// CHECK-LABEL: define internal float @__block_in_function_block_invoke
// CHECK-SAME:  #[[ATTR:[0-9]+]]
// CHECK:       call float @llvm.experimental.constrained.fadd.f32({{.*}}, metadata !"round.dynamic", metadata !"fpexcept.strict")
float block_in_function(float x, float y) {
  float (^blk)(float, float) = ^float(float a, float b) { return a + b; };
  return blk(x, y);
}

// CHECK: attributes #[[ATTR]] = {{.*}} strictfp
