// RUN: %clang_cc1 -emit-llvm-bc -disable-llvm-passes -o %t.bc %s
// RUN: llvm-dis %t.bc -o - | FileCheck %s

// Test case for PR45426. Make sure we do not crash while writing bitcode
// containing a simplify-able fneg constant expression. Check that the created
// bitcode file can be disassembled and has the constant expressions simplified.
//
// CHECK-LABEL define i32 @main()
// CHECK:      entry:
// CHECK-NEXT:   %retval = alloca i32
// CHECK-NEXT:   store i32 0, i32* %retval
// CHECK-NEXT:   [[LV:%.*]] = load float*, float** @c
// CHECK-NEXT:   store float 1.000000e+00, float* [[LV]], align 4
// CHECK-NEXT:   ret i32 -1

int a[], b;
float *c;
int main() {
  return -(*c = &b != a);
}
