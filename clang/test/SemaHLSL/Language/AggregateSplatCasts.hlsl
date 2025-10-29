// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -finclude-default-header -fnative-half-type %s -ast-dump | FileCheck %s

// splat from vec1 to vec
// CHECK-LABEL: call1
// CHECK: CStyleCastExpr {{.*}} 'int3':'vector<int, 3>' <HLSLAggregateSplatCast>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <FloatingToIntegral> part_of_explicit_cast
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <HLSLVectorTruncation> part_of_explicit_cast
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float1':'vector<float, 1>' <LValueToRValue> part_of_explicit_cast
// CHECK-NEXT: DeclRefExpr {{.*}} 'float1':'vector<float, 1>' lvalue Var {{.*}} 'A' 'float1':'vector<float, 1>'
export void call1() {
  float1 A = {1.0};
  int3 B = (int3)A;
}

struct S {
  int A;
  float B;
  int C;
  float D;
};

// splat from scalar to aggregate
// CHECK-LABEL: call2
// CHECK: CStyleCastExpr {{.*}} 'S' <HLSLAggregateSplatCast>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 5
export void call2() {
  S s = (S)5; 
}
