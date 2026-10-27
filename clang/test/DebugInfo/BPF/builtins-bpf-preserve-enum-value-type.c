// REQUIRES: bpf-registered-target
// RUN: %clang_cc1 -triple bpf -emit-llvm -debug-info-kind=limited -disable-llvm-passes %s -o - | FileCheck %s

// __builtin_preserve_enum_value() returns the type of the enumerator, which is
// the underlying type for an enum with a fixed underlying type. The i64 result
// of the intrinsic is converted to it, so the value compares the same as the
// enumerator.

#define _(x) __builtin_preserve_enum_value((x), 1)

enum E1 : unsigned { E1_A = 4 };
enum E2 : signed { E2_A = -4 };
enum E3 : unsigned long { E3_A = 4 };
enum E4 : long { E4_A = -4 };
enum E5 { E5_A = 4 };
enum E6 : _Bool { E6_A, E6_B };
enum E7 { E7_A = 4, E7_B = 0x100000000 };

int unit1(unsigned x) { return x < _(*(enum E1 *)E1_A); }

// CHECK-LABEL: define {{.*}}i32 @unit1
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 0, ptr @0, i64 1)
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: icmp ult i32 %{{[0-9a-z]+}}, [[T]]

int unit2(int x) { return x < _(*(enum E2 *)E2_A); }

// CHECK-LABEL: define {{.*}}i32 @unit2
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 1, ptr @1, i64 1)
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: icmp slt i32 %{{[0-9a-z]+}}, [[T]]

int unit3(unsigned long x) { return x < _(*(enum E3 *)E3_A); }

// CHECK-LABEL: define {{.*}}i32 @unit3
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 2, ptr @2, i64 1)
// CHECK-NEXT: icmp ult i64 %{{[0-9a-z]+}}, [[V]]

int unit4(long x) { return x < _(*(enum E4 *)E4_A); }

// CHECK-LABEL: define {{.*}}i32 @unit4
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 3, ptr @3, i64 1)
// CHECK-NEXT: icmp slt i64 %{{[0-9a-z]+}}, [[V]]

int unit5(long x) { return x < _(*(enum E5 *)E5_A); }

// CHECK-LABEL: define {{.*}}i32 @unit5
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 4, ptr @4, i64 1)
// CHECK-NEXT: [[T:%[0-9a-z]+]] = trunc i64 [[V]] to i32
// CHECK-NEXT: [[S:%[0-9a-z]+]] = sext i32 [[T]] to i64
// CHECK-NEXT: icmp slt i64 %{{[0-9a-z]+}}, [[S]]

// For _Bool, the value is converted with != 0 rather than truncated.
int unit6(int x) { return x == _(*(enum E6 *)E6_B); }

// CHECK-LABEL: define {{.*}}i32 @unit6
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 5, ptr @5, i64 1)
// CHECK-NEXT: [[B:%[0-9a-z]+]] = icmp ne i64 [[V]], 0
// CHECK-NEXT: [[Z:%[0-9a-z]+]] = zext i1 [[B]] to i32
// CHECK-NEXT: icmp eq i32 %{{[0-9a-z]+}}, [[Z]]

// An enumerator of an enum with a 64-bit value has type unsigned long.
int unit7(unsigned long x) { return x < _(*(enum E7 *)E7_A); }

// CHECK-LABEL: define {{.*}}i32 @unit7
// CHECK: [[V:%[0-9a-z]+]] = call i64 @llvm.bpf.preserve.enum.value(i32 6, ptr @6, i64 1)
// CHECK-NEXT: icmp ult i64 %{{[0-9a-z]+}}, [[V]]
