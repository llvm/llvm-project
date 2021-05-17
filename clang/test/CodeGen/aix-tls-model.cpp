// RUN: %clang_cc1 %s -triple powerpc-unknown-aix -target-cpu pwr8 -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-GD
// RUN: %clang_cc1 %s -triple powerpc-unknown-aix -target-cpu pwr8 -ftls-model=global-dynamic -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-GD
// RUN: not %clang_cc1 %s -triple powerpc-unknown-aix -target-cpu pwr8 -ftls-model=local-dynamic -emit-llvm 2>&1 | FileCheck %s -check-prefix=CHECK-LD-ERROR
// RUN: not %clang_cc1 %s -triple powerpc-unknown-aix -target-cpu pwr8 -ftls-model=initial-exec -emit-llvm  2>&1 | FileCheck %s -check-prefix=CHECK-IE-ERROR
// RUN: not %clang_cc1 %s -triple powerpc-unknown-aix -target-cpu pwr8 -ftls-model=local-exec -emit-llvm 2>&1 | FileCheck %s -check-prefix=CHECK-LE-ERROR
// RUN: %clang_cc1 %s -triple powerpc64-unknown-aix -target-cpu pwr8 -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-GD
// RUN: %clang_cc1 %s -triple powerpc64-unknown-aix -target-cpu pwr8 -ftls-model=global-dynamic -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-GD
// RUN: not %clang_cc1 %s -triple powerpc64-unknown-aix -target-cpu pwr8 -ftls-model=local-dynamic -emit-llvm 2>&1 | FileCheck %s -check-prefix=CHECK-LD-ERROR
// RUN: not %clang_cc1 %s -triple powerpc64-unknown-aix -target-cpu pwr8 -ftls-model=initial-exec -emit-llvm  2>&1 | FileCheck %s -check-prefix=CHECK-IE-ERROR
// RUN: not %clang_cc1 %s -triple powerpc64-unknown-aix -target-cpu pwr8 -ftls-model=local-exec -emit-llvm 2>&1 | FileCheck %s -check-prefix=CHECK-LE-ERROR

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
// CHECK-IE-ERROR:  error: TLS model 'initial-exec' is not yet supported on AIX
// CHECK-LE-ERROR:  error: TLS model 'local-exec' is not yet supported on AIX
