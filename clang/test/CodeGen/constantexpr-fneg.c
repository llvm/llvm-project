// RUN: %clang_cc1 -no-opaque-pointers -emit-llvm-bc -disable-llvm-passes -o %t.bc %s
// RUN: llvm-dis %t.bc -o - | FileCheck %s

// Test case for PR45426. Make sure we do not crash while writing bitcode
// containing a simplify-able fneg constant expression.
//
// CHECK-LABEL define i32 @main()
// CHECK:      entry:
// CHECK-NEXT:   %retval = alloca i32
// CHECK-NEXT:   store i32 0, i32* %retval
// CHECK-NEXT:   [[LV:%.*]] = load float*, float** @c
// CHECK-NEXT:   store float 1.000000e+00, float* [[LV]], align 4
// CHECK-NEXT:   [[FNEG:%.*]] = fneg float 1.000000e+00
// CHECK-NEXT:   [[CONV:%.*]] = fptosi float [[FNEG]] to i32
// CHECK-NEXT:   ret i32 [[CONV]]

int a[], b;
float *c;
int main(void) {
  return -(*c = &b != a);
}
