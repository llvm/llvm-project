// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s

float fff(float x, float y) {
// CHECK-LABEL: define float @_Z3fffff{{.*}}
// CHECK: entry
#pragma float_control(except, on)
  float z;
  z = z * z;
  //CHECK: llvm.experimental.constrained.fmul{{.*}}
  {
    z = x * y;
    //CHECK: llvm.experimental.constrained.fmul{{.*}}
  }
  {
// This pragma has no effect since if there are any fp intrin in the
// function then all the operations need to be fp intrin
#pragma float_control(except, off)
    z = z + x * y;
    //CHECK: llvm.experimental.constrained.fmul{{.*}}
  }
  z = z * z;
  //CHECK: llvm.experimental.constrained.fmul{{.*}}
  return z;
}
float check_precise(float x, float y) {
  // CHECK-LABEL: define float @_Z13check_preciseff{{.*}}
  float z;
  {
#pragma float_control(precise, on)
    z = x * y + z;
    //CHECK: llvm.fmuladd{{.*}}
  }
  {
#pragma float_control(precise, off)
    z = x * y + z;
    //CHECK: fmul fast float
    //CHECK: fadd fast float
  }
  return z;
}
float fma_test1(float a, float b, float c) {
// CHECK-LABEL define float @_Z9fma_test1fff{{.*}}
#pragma float_control(precise, on)
  float x = a * b + c;
  //CHECK: fmuladd
  return x;
}
