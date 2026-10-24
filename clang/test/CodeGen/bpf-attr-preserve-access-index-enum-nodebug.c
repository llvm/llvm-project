// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s

// Without -g, the enumerators of an enum with the preserve_access_index
// attribute are not relocated, and their values are used. They are still not
// constant expressions, as with -g, see
// clang/test/Sema/bpf-preserve-enum-value-parity.c.
enum e1 { E1_A = 1, E1_C = 10 } __attribute__((preserve_access_index));

int test(int x) {
  if (x == E1_C)
    return E1_A;
  return 0;
}

// CHECK: define dso_local i32 @test
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: icmp eq i32 %{{[0-9a-z]+}}, 10
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: store i32 1, ptr %retval
// CHECK-NOT: @llvm.bpf.preserve.enum.value
// CHECK: ret i32
