// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-GD
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=global-dynamic -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-GD
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=local-dynamic -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-LD
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=initial-exec -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-IE
// RUN: %clang_cc1 %s -triple x86_64-pc-linux-gnu -ftls-model=local-exec -emit-llvm -o - | FileCheck %s -check-prefix=CHECK-LE

int __thread x;
int f() {
  static int __thread y;
  return y++;
}
int __thread __attribute__((tls_model("initial-exec"))) z;

// CHECK-GD: @f.y = internal thread_local global i32 0
// CHECK-GD: @x = thread_local global i32 0
// CHECK-GD: @z = thread_local(initialexec) global i32 0

// CHECK-LD: @f.y = internal thread_local(localdynamic) global i32 0
// CHECK-LD: @x = thread_local(localdynamic) global i32 0
// CHECK-LD: @z = thread_local(initialexec) global i32 0

// CHECK-IE: @f.y = internal thread_local(initialexec) global i32 0
// CHECK-IE: @x = thread_local(initialexec) global i32 0
// CHECK-IE: @z = thread_local(initialexec) global i32 0

// CHECK-LE: @f.y = internal thread_local(localexec) global i32 0
// CHECK-LE: @x = thread_local(localexec) global i32 0
// CHECK-LE: @z = thread_local(initialexec) global i32 0
