// RUN: %clang_cc1 -triple i686-pc-linux-gnu -std=c2x -emit-llvm -o - %s | FileCheck %s

// Ensure that thread_local and _Thread_local emit the same codegen. See
// https://github.com/llvm/llvm-project/issues/70068 for details.

void func(void) {
  static thread_local int i = 12;
  static _Thread_local int j = 13;

  extern thread_local int k;
  extern thread_local int l;

  (void)k;
  (void)l;
}

// CHECK:      @func.i = internal thread_local global i32 12, align 4
// CHECK-NEXT: @func.j = internal thread_local global i32 13, align 4
// CHECK-NEXT: @k = external thread_local global i32, align 4
// CHECK-NEXT: @l = external thread_local global i32, align 4

// CHECK:      define dso_local void @func()
// CHECK-NEXT: entry:
// CHECK-NEXT: %[[K:.+]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @k)
// CHECK-NEXT: load i32, ptr %[[K]], align 4
// CHECK-NEXT: %[[L:.+]] = call align 4 ptr @llvm.threadlocal.address.p0(ptr align 4 @l)
// CHECK-NEXT: load i32, ptr %[[L]], align 4
