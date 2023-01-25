// RUN: %clang_cc1 -triple x86_64-apple-macosx10.7.0 %s -emit-llvm -o - | FileCheck %s

void t1(void) __attribute__((naked));

// Basic functionality check
// (Note that naked needs to imply noinline to work properly.)
// CHECK: define{{.*}} void @t1() [[NAKED_OPTNONE:#[0-9]+]] {
void t1(void)
{
}

// Make sure this doesn't explode in the verifier.
// (It doesn't really make sense, but it isn't invalid.)
// CHECK: define{{.*}} void @t2() [[NAKED:#[0-9]+]] {
__attribute((naked, always_inline)) void t2(void) {
}

// Make sure not to generate prolog or epilog for naked functions.
__attribute((naked)) void t3(int x) {
// CHECK: define{{.*}} void @t3(i32 noundef %0)
// CHECK-NOT: alloca
// CHECK-NOT: store
// CHECK: unreachable
}

// Make sure naked functions do not attempt to evaluate parameters with a
// variably-modified type. Naked functions get no prolog, so this evaluation
// should not take place.
__attribute__((naked)) void t4(int len, char x[len]) {
  // CHECK: define{{.*}} void @t4(i32 noundef{{.*}}, ptr noundef{{.*}})
  // CHECK: unreachable
}

// CHECK: attributes [[NAKED_OPTNONE]] = { naked noinline nounwind optnone{{.*}} }
// CHECK: attributes [[NAKED]] = { naked noinline nounwind{{.*}} }
