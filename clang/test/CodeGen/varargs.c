// RUN: %clang_cc1 -triple i386-unknown-unknown -emit-llvm -o - %s | FileCheck %s

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
