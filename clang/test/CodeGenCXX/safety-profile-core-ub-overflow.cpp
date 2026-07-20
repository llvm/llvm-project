// RUN: %clang_cc1 -std=c++23 -fprofiles -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF

// std::core_ub / {expr.mul.representable.type.result} (P4317 A.1): a signed
// arithmetic result that is not representable is undefined; under enforcement
// the operation checks for overflow and traps. Unsigned arithmetic wraps and
// is not guarded.

[[profiles::enforce(std::core_ub)]];

// CHECK-LABEL: define {{.*}}@_Z3addii
// CHECK: @llvm.sadd.with.overflow.i32
// CHECK: call void @llvm.ubsantrap(i8 0)
// OFF-LABEL: define {{.*}}@_Z3addii
// OFF-NOT: llvm.ubsantrap
int add(int a, int b) { return a + b; }

// CHECK-LABEL: define {{.*}}@_Z3mulii
// CHECK: @llvm.smul.with.overflow.i32
// CHECK: call void @llvm.ubsantrap(i8 12)
int mul(int a, int b) { return a * b; }

// CHECK-LABEL: define {{.*}}@_Z6negatei
// CHECK: @llvm.ssub.with.overflow.i32
// CHECK: call void @llvm.ubsantrap(i8 13)
int negate(int a) { return -a; }

// Unsigned arithmetic is defined to wrap, so it is never guarded.
// CHECK-LABEL: define {{.*}}@_Z4uaddjj
// CHECK-NOT: llvm.ubsantrap
// CHECK: {{^}}}
unsigned uadd(unsigned a, unsigned b) { return a + b; }
