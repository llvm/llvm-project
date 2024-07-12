// RUN: %clang_cc1 -emit-llvm -std=c99 %s -o - | FileCheck %s

// Demonstrate that statics are properly zero initialized.
static _Complex float f_global;
void func(void) {
  static _Complex double d_local;
  d_local = f_global;
}

// CHECK-DAG: @func.d_local = internal global { double, double } zeroinitializer
// CHECK-DAG: @f_global = internal global { float, float } zeroinitializer

