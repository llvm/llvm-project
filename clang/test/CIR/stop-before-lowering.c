// -clangir-stop-before-lowering halts the pipeline at the ABI-free boundary:
// cleanup passes run, but target/ABI lowering does not. The struct is still
// returned by value. Without the flag, CallConvLowering coerces the return.
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:   -clangir-stop-before-lowering -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=STOP
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o - \
// RUN:   | FileCheck %s --check-prefix=LOWERED

struct S {
  int a, b;
};

struct S make(int x) {
  struct S s = {x, x};
  return s;
}

// STOP: cir.func {{.*}}@make(%arg0: !s32i {{.*}}) -> !rec_S
// LOWERED-NOT: @make{{.*}} -> !rec_S
