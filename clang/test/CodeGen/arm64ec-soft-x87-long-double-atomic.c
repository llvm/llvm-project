// RUN: %clang_cc1 -triple arm64ec-w64-windows-gnu -emit-llvm -o - %s | FileCheck %s

// The ABI moves a `long double` as a 16-byte integer image, but the value
// itself is a real `x86_fp80`; the read-modify-write therefore loads and
// stores the image and performs the increment as a floating-point operation.
_Atomic long double g;

long double pre_inc(void) { return ++g; }
// CHECK-LABEL: define {{.*}} @pre_inc(
// CHECK-NOT: atomicrmw
// CHECK: load atomic i128, ptr @g seq_cst
// CHECK: [[TMP:%.*]] = load x86_fp80, ptr %atomic-temp
// CHECK: [[OLD:%.*]] = phi x86_fp80 [ [[TMP]], %entry ]
// CHECK: fadd x86_fp80 [[OLD]], 1.000000e+00
// CHECK: cmpxchg ptr @g, i128 {{.*}}, i128 {{.*}} seq_cst seq_cst, align 16

long double post_dec(void) { return g--; }
// CHECK-LABEL: define {{.*}} @post_dec(
// CHECK-NOT: atomicrmw
// CHECK: load atomic i128, ptr @g seq_cst
// CHECK: fadd x86_fp80 {{.*}}, -1.000000e+00
// CHECK: cmpxchg ptr @g, i128 {{.*}}, i128 {{.*}} seq_cst seq_cst, align 16

void add_assign(void) { g += 1.0L; }
// CHECK-LABEL: define {{.*}} @add_assign(
// CHECK-NOT: atomicrmw
// CHECK: cmpxchg ptr @g, i128 {{.*}}, i128 {{.*}} seq_cst seq_cst, align 16
