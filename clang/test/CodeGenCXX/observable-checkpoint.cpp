// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++11 -O2 -Wno-return-type -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++26 -O2 -Wno-return-type -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++26 -O3 -ffreestanding -Wno-return-type -emit-llvm -o - %s | FileCheck %s

// P1494R5 and P3641R0: partial program correctness.

extern int x;

// Adapted from GCC's g++.dg/cpp26/observable-checkpoint.C. Falling off the
// end of this function is undefined behavior, including after inlining.
int here_b_ub() {}

// CHECK-LABEL: define {{.*}}void @gcc_example(
// CHECK: store i32 0, ptr @x
// CHECK: call i32 {{.*}}@{{puts|printf}}(
// CHECK: [[X:%.*]] = load i32, ptr @x
// CHECK: [[ADD:%.*]] = add nsw i32 [[X]], 42
// CHECK: store i32 [[ADD]], ptr @x
// CHECK: call void @llvm.observable.checkpoint()
// CHECK-NEXT: unreachable
extern "C" void gcc_example() {
  x = 0;
  __builtin_printf(" start \n");
  x += 42;
  __builtin_observable_checkpoint();
  here_b_ub();
}

// CHECK-LABEL: define {{.*}}void @checkpoint(
// CHECK: call void @llvm.observable.checkpoint()
// CHECK-NEXT: ret void
extern "C" __attribute__((always_inline)) void checkpoint() noexcept {
  __builtin_observable_checkpoint();
}

// Inlining a library wrapper must preserve the checkpoint.
// CHECK-LABEL: define {{.*}}void @inlined_checkpoint(
// CHECK: call void @llvm.observable.checkpoint()
// CHECK-NEXT: unreachable
extern "C" void inlined_checkpoint() {
  checkpoint();
  __builtin_unreachable();
}

// The return from an I/O call is an implicit checkpoint. I/O library calls
// already model possible termination, so they protect their own output as well
// as earlier observable behavior without an explicit intrinsic.
// CHECK-LABEL: define {{.*}}void @implicit_checkpoint(
// CHECK: icmp eq ptr {{.*}}, null
// CHECK: br i1
// CHECK: call i32 {{.*}}@{{puts|printf}}(
// CHECK: load i32, ptr
// CHECK: ret void
extern "C" void implicit_checkpoint(int *p) {
  if (!p)
    __builtin_printf("null pointer\n");
  ++*p;
}

// CHECK-LABEL: define {{.*}}void @io_before_ub(
// CHECK: call i32 {{.*}}@{{puts|printf}}(
// CHECK-NEXT: unreachable
extern "C" void io_before_ub() {
  __builtin_printf("defined prefix\n");
  here_b_ub();
}
