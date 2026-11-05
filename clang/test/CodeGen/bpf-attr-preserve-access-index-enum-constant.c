// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -debug-info-kind=limited -emit-llvm -disable-llvm-passes \
// RUN:   -Wno-bpf-preserve-access-index-enumerator %s -o - | FileCheck %s

// With -Wno-bpf-preserve-access-index-enumerator, an enumerator of an enum with
// the preserve_access_index attribute is a constant where a constant is
// required, and its value in the local enum is used. Elsewhere, it is still
// relocated.
enum e1 { E1_A = 1, E1_B = 2 } __attribute__((preserve_access_index));

struct ops { unsigned long flags; };

// CHECK-DAG: @arr = global [2 x i32] zeroinitializer
int arr[E1_A + 1];
// CHECK-DAG: @ops = global %struct.ops { i64 2 }
struct ops ops = { .flags = E1_B };

int test(int x) {
  asm volatile("r0 = %0" :: "i"(E1_B) : "r0");
  return x == E1_B;
}

// CHECK-LABEL: define dso_local i32 @test(
// CHECK: call void asm sideeffect "r0 = $0", "i,~{r0}"(i32 2)
// CHECK: call i64 @llvm.bpf.preserve.enum.value(
