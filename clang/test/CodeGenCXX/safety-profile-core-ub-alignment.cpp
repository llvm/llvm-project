// RUN: %clang_cc1 -std=c++23 -fprofiles -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF

// std::core_ub / {basic.align.object.alignment} (P4317 A.1): accessing an
// object through a pointer that does not meet the type's alignment is
// undefined; under enforcement the access checks the pointer's low bits and
// traps. The int load requires 4-byte alignment, so the check masks the low
// two bits.

[[profiles::enforce(std::core_ub)]];

// CHECK-LABEL: define {{.*}}@_Z8load_intPv
// CHECK: and i64 %{{.*}}, 3
// CHECK: call void @llvm.ubsantrap(i8 22)
// OFF-LABEL: define {{.*}}@_Z8load_intPv
// OFF-NOT: llvm.ubsantrap
int load_int(void *p) { return *static_cast<int *>(p); }
