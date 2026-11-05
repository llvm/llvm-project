// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -fblocks -debug-info-kind=limited -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

// A block literal in the operand of sizeof is parsed in a potentially evaluated
// context nested in the unevaluated one. The enumerator stays a constant there
// at first. Since the operand has a variably modified type, it is rebuilt as
// potentially evaluated, which relocates the enumerator.
enum e1 { E1_A = 4 } __attribute__((preserve_access_index));

// CHECK: @[[E1_A:[0-9]+]] = private unnamed_addr constant [7 x i8] c"E1_A:4\00"

unsigned long test1(int n) {
  return sizeof(*(int (*)[n])(^{ return (char *)0 + E1_A; })());
}

// CHECK: define internal ptr @__test1_block_invoke(
// CHECK: call i64 @llvm.bpf.preserve.enum.value(i32 {{[0-9]+}}, ptr @[[E1_A]], i64 1)
