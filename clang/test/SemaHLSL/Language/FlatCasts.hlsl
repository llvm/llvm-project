// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -fnative-half-type %s -ast-dump | FileCheck %s

// truncation
// CHECK-LABEL: call1
// CHECK: CStyleCastExpr {{.*}} 'int[1]' <HLSLElementwiseCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int[2]' <HLSLArrayRValue> part_of_explicit_cast
// CHECK-NEXT: DeclRefExpr {{.*}} 'int[2]' lvalue Var {{.*}} 'A' 'int[2]'
export void call1() {
  int A[2] = {0,1};
  int B[1] = {4};
  B = (int[1])A;
}

// flat cast of equal size
// CHECK-LABEL: call2
// CHECK: CStyleCastExpr {{.*}} 'float[1]' <HLSLElementwiseCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int[1]' <HLSLArrayRValue> part_of_explicit_cast
// CHECK-NEXT: DeclRefExpr {{.*}} 'int[1]' lvalue Var {{.*}} 'A' 'int[1]'
export void call2() {
  int A[1] = {0};
  float B[1] = {1.0};
  B = (float[1])A;
}
