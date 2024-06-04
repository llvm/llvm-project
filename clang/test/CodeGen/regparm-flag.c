// RUN: %clang_cc1 -triple i386-unknown-unknown -mregparm 4 %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -triple i386-unknown-unknown -fsanitize=array-bounds %s -emit-llvm -o - | FileCheck %s --check-prefix=RUNTIME0
// RUN: %clang_cc1 -triple i386-unknown-unknown -mregparm 1 -fsanitize=array-bounds %s -emit-llvm -o - | FileCheck %s --check-prefix=RUNTIME1
// RUN: %clang_cc1 -triple i386-unknown-unknown -mregparm 2 -fsanitize=array-bounds %s -emit-llvm -o - | FileCheck %s --check-prefix=RUNTIME2
// RUN: %clang_cc1 -triple i386-unknown-unknown -mregparm 3 -fsanitize=array-bounds %s -emit-llvm -o - | FileCheck %s --check-prefix=RUNTIME2

void f1(int a, int b, int c, int d,
        int e, int f, int g, int h);

void f2(int a, int b) __attribute((regparm(0)));

void f0(void) {
// CHECK: call void @f1(i32 inreg noundef 1, i32 inreg noundef 2, i32 inreg noundef 3, i32 inreg noundef 4,
// CHECK: i32 noundef 5, i32 noundef 6, i32 noundef 7, i32 noundef 8)
  f1(1, 2, 3, 4, 5, 6, 7, 8);
// CHECK: call void @f2(i32 noundef 1, i32 noundef 2)
  f2(1, 2);
}

struct has_array {
  int a;
  int b[4];
  int c;
};

int access(struct has_array *p, int index)
{
  return p->b[index];
}

// CHECK: declare void @f1(i32 inreg noundef, i32 inreg noundef, i32 inreg noundef, i32 inreg noundef,
// CHECK: i32 noundef, i32 noundef, i32 noundef, i32 noundef)
// CHECK: declare void @f2(i32 noundef, i32 noundef)

// RUNTIME0: declare void @__ubsan_handle_out_of_bounds_abort(ptr, i32)
// RUNTIME1: declare void @__ubsan_handle_out_of_bounds_abort(ptr inreg, i32)
// RUNTIME2: declare void @__ubsan_handle_out_of_bounds_abort(ptr inreg, i32 inreg)
