// RUN: %clang_cc1 -triple avr-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// The depth argument of llvm.frameaddress and llvm.returnaddress is always a
// 32-bit integer. It used to be emitted with the type of 'unsigned int', which
// is only 16 bits wide on AVR, producing an intrinsic call with a bad
// signature and crashing clang.

// CHECK-LABEL: define{{.*}} ptr @frame_address_zero(
// CHECK: call{{.*}}@llvm.frameaddress.p0(i32 0)
void *frame_address_zero(void) { return __builtin_frame_address(0); }

// CHECK-LABEL: define{{.*}} ptr @return_address_zero(
// CHECK: call{{.*}}@llvm.returnaddress.p1(i32 0)
void *return_address_zero(void) { return __builtin_return_address(0); }

// CHECK-LABEL: define{{.*}} ptr @frame_address_depth(
// CHECK: call{{.*}}@llvm.frameaddress.p0(i32 2)
void *frame_address_depth(void) { return __builtin_frame_address(2); }

// CHECK-LABEL: define{{.*}} ptr @return_address_depth(
// CHECK: call{{.*}}@llvm.returnaddress.p1(i32 2)
void *return_address_depth(void) { return __builtin_return_address(2); }
