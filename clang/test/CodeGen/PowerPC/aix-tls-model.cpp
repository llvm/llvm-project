// RUN: %clang_cc1 %s -triple powerpc-unknown-aix -target-cpu pwr8 -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-GD
// RUN: %clang_cc1 %s -triple powerpc-unknown-aix -target-cpu pwr8 -ftls-model=global-dynamic -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-GD
// RUN: not %clang_cc1 %s -triple powerpc-unknown-aix -target-cpu pwr8 -ftls-model=local-dynamic -emit-llvm 2>&1 | FileCheck %s -check-prefix=CHECK-LD-ERROR
// RUN: %clang_cc1 %s -triple powerpc-unknown-aix -target-cpu pwr8 -ftls-model=initial-exec -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-IE
// RUN: %clang_cc1 %s -triple powerpc-unknown-aix -target-cpu pwr8 -ftls-model=local-exec -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-LE
// RUN: %clang_cc1 %s -triple powerpc64-unknown-aix -target-cpu pwr8 -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-GD
// RUN: %clang_cc1 %s -triple powerpc64-unknown-aix -target-cpu pwr8 -ftls-model=global-dynamic -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-GD
// RUN: not %clang_cc1 %s -triple powerpc64-unknown-aix -target-cpu pwr8 -ftls-model=local-dynamic -emit-llvm 2>&1 | FileCheck %s -check-prefix=CHECK-LD-ERROR
// RUN: %clang_cc1 %s -triple powerpc64-unknown-aix -target-cpu pwr8 -ftls-model=initial-exec -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-IE
// RUN: %clang_cc1 %s -triple powerpc64-unknown-aix -target-cpu pwr8 -ftls-model=local-exec -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-LE

int z1 = 0;
int z2;
int __thread x;
int f() {
  static int __thread y;
  return y++;
}

// CHECK-GD: @z1 ={{.*}} global i32 0
// CHECK-GD: @z2 ={{.*}} global i32 0
// CHECK-GD: @x ={{.*}} thread_local global i32 0
// CHECK-GD: @_ZZ1fvE1y = internal thread_local global i32 0
// CHECK-LD-ERROR:  error: TLS model 'local-dynamic' is not yet supported on AIX
// CHECK-IE: @z1 ={{.*}} global i32 0
// CHECK-IE: @z2 ={{.*}} global i32 0
// CHECK-IE: @x ={{.*}} thread_local(initialexec) global i32 0
// CHECK-IE: @_ZZ1fvE1y = internal thread_local(initialexec) global i32 0
// CHECK-LE: @z1 ={{.*}} global i32 0
// CHECK-LE: @z2 ={{.*}} global i32 0
// CHECK-LE: @x ={{.*}} thread_local(localexec) global i32 0
// CHECK-LE: @_ZZ1fvE1y = internal thread_local(localexec) global i32 0
