// RUN: %clang_cc1 -triple i386-unknown-unknown -std=c2y -emit-llvm -o - %s | FileCheck %s

// PR6433 - Don't crash on va_arg(typedef).
typedef double gdouble;
void focus_changed_cb (void) {
    __builtin_va_list pa;
    double mfloat;
    mfloat = __builtin_va_arg((pa), gdouble);
}

void vararg(int, ...);
void function_as_vararg(void) {
  // CHECK: define {{.*}}function_as_vararg
  // CHECK-NOT: llvm.trap
  vararg(0, focus_changed_cb);
}

void vla(int n, ...)
{
  __builtin_va_list ap;
  void *p;
  p = __builtin_va_arg(ap, typeof (int (*)[++n])); // CHECK: add nsw i32 {{.*}}, 1
}

// Ensure that __builtin_va_start(list, 0) and __builtin_c23_va_start(list)
// have the same codegen.
void noargs(...) {
  __builtin_va_list list;
  // CHECK:   %list = alloca ptr
  __builtin_va_start(list, 0);
  // CHECK-NEXT: call void @llvm.va_start.p0(ptr %list)  
  __builtin_c23_va_start(list);
  // CHECK-NEXT: call void @llvm.va_start.p0(ptr %list)
}
