// Stores through a pointer-valued conditional or comma: the two shapes named by
// the "TODO: conditional operators, comma" in EmitPointerWithAlignment. The
// requirement is lost there, so neither is rejected. This file asserts the
// answer they should get, so implementing that TODO will make it pass and the
// XFAIL can go.
//
// XFAIL: *
// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm -fsanitize=array-bounds \
// RUN:     -Wno-array-bounds -std=c11 %s -o - | FileCheck %s

int a[4];

// CHECK-LABEL: define {{.*}}@cond_arm(
// CHECK: icmp ult i64 {{.*}}, 4
void cond_arm(int i, int c) { *(c ? &a[i] : &a[0]) = 1; }

// CHECK-LABEL: define {{.*}}@comma_addr(
// CHECK: icmp ult i64 {{.*}}, 4
void comma_addr(int i) { *(1, &a[i]) = 1; }
