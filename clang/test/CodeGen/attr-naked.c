// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7.0 %s -emit-llvm -o - | FileCheck %s

void t1() __attribute__((naked));

// Basic functionality check
// (Note that naked needs to imply noinline to work properly.)
// CHECK: define{{.*}} void @t1() [[NAKED_OPTNONE:#[0-9]+]] {
void t1()
{
}

// Make sure this doesn't explode in the verifier.
// (It doesn't really make sense, but it isn't invalid.)
// CHECK: define{{.*}} void @t2() [[NAKED:#[0-9]+]] {
__attribute((naked, always_inline)) void t2() {
}

// Make sure not to generate prolog or epilog for naked functions.
__attribute((naked)) void t3(int x) {
// CHECK: define{{.*}} void @t3(i32 %0)
// CHECK-NOT: alloca
// CHECK-NOT: store
// CHECK: unreachable
}

// CHECK: attributes [[NAKED_OPTNONE]] = { naked noinline nounwind optnone{{.*}} }
// CHECK: attributes [[NAKED]] = { naked noinline nounwind{{.*}} }
