// RUN: %clang_cc1 -std=c++23 -fprofiles -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -std=c++23 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF

// std::core_ub / {expr.add.out.of.bounds} with a statically known bound
// (P4317 A.1): indexing past the end of an array whose extent is visible at
// the subscript is undefined; under enforcement the subscript checks the index
// against the bound and traps. The array-reference parameter keeps the extent
// visible.

[[profiles::enforce(std::core_ub)]];

// CHECK-LABEL: define {{.*}}@_Z3getRA4_ii
// CHECK: icmp ult i64 %{{.*}}, 4
// CHECK: call void @llvm.ubsantrap(i8 18)
// OFF-LABEL: define {{.*}}@_Z3getRA4_ii
// OFF-NOT: llvm.ubsantrap
int get(int (&a)[4], int i) { return a[i]; }
