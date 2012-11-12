// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s


// PR6433 - Don't crash on va_arg(typedef).
typedef double gdouble;
void focus_changed_cb () {
    __builtin_va_list pa;
    double mfloat;
    mfloat = __builtin_va_arg((pa), gdouble);
}

void vararg(int, ...);
void function_as_vararg() {
  // CHECK: define {{.*}}function_as_vararg
  // CHECK-NOT: llvm.trap
  vararg(0, focus_changed_cb);
}
