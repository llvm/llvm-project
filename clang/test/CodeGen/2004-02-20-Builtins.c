// RUN: %clang_cc1  %s -emit-llvm -o - | FileCheck %s
double sqrt(double x);

// CHECK-LABEL: @zsqrtxxx
// CHECK-NOT: builtin
void zsqrtxxx(float num) {
   num = sqrt(num);
}
