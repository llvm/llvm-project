// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -fblocks -debug-info-kind=limited -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

// An enumerator in a block literal in sizeof is relocated only when the
// variably modified operand is rebuilt as potentially evaluated.
enum e1 { E1_A = 4 } __attribute__((preserve_access_index));

// CHECK: @[[E1_A:[0-9]+]] = private unnamed_addr constant [7 x i8] c"E1_A:4\00"

unsigned long test1(int n) {
  return sizeof(*(int (*)[n])(^{ return (char *)0 + E1_A; })());
}

// CHECK: define internal ptr @__test1_block_invoke(
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[E1_A]], i64 1)

// A static local variable needs a constant initializer, and a block literal is
// one. Its body runs only when the block is called, so the enumerator there is
// still relocated.
int test2(void) {
  static int (^b)(void) = ^{ return (int)E1_A; };
  return b();
}

// CHECK: define internal i32 @_block_invoke(
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @{{[0-9]+}}, i64 1)
