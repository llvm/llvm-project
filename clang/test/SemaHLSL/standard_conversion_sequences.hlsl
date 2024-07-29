// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -Wconversion -verify -o - %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -Wno-conversion -DNO_ERR -ast-dump %s | FileCheck %s

void test() {

  // CHECK: VarDecl {{.*}} used f3 'vector<float, 3>' cinit
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 3>' <VectorSplat>
  // CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
  vector<float,3> f3 = 1.0; // No warning for splatting to a vector from a literal.


  // CHECK: VarDecl {{.*}} used d4 'vector<double, 4>' cinit
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<double, 4>' <FloatingCast>
  // CHECK-NEXT: ExtVectorElementExpr {{.*}} 'vector<float, 4>' xyzx
  // CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, 3>' lvalue Var {{.*}} 'f3' 'vector<float, 3>'
  vector<double,4> d4 = f3.xyzx; // No warnings for promotion or explicit extension.

  // CHECK: VarDecl {{.*}} used f2 'vector<float, 2>' cinit
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 2>' <HLSLVectorTruncation>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 3>' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, 3>' lvalue Var {{.*}} 'f3' 'vector<float, 3>'
  // expected-warning@#f2{{implicit conversion truncates vector: 'vector<float, 3>' (vector of 3 'float' values) to 'vector<float, 2>' (vector of 2 'float' values)}}
  vector<float,2> f2 = f3; // #f2

  // CHECK: VarDecl {{.*}} f2_2 'vector<float, 2>' cinit
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 2>' <HLSLVectorTruncation>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 4>' <FloatingCast>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<double, 4>' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'vector<double, 4>' lvalue Var {{.*}} 'd4' 'vector<double, 4>'
  // expected-warning@#f2_2{{implicit conversion truncates vector: 'vector<double, 4>' (vector of 4 'double' values) to 'vector<float, 2>' (vector of 2 'float' values)}}
  // expected-warning@#f2_2{{implicit conversion loses floating-point precision: 'vector<double, 4>' (vector of 4 'double' values) to 'vector<float, 2>' (vector of 2 'float' values)}}
  vector<float,2> f2_2 = d4; // #f2_2

  // CHECK: VarDecl {{.*}} i2 'vector<int, 2>' cinit
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 2>' <FloatingToIntegral>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<float, 2>' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'vector<float, 2>' lvalue Var {{.*}} 'f2' 'vector<float, 2>'
  // expected-warning@#i2{{mplicit conversion turns floating-point number into integer: 'vector<float, 2>' (vector of 2 'float' values) to 'vector<int, 2>' (vector of 2 'int' values)}}
  vector<int,2> i2 = f2; // #i2

  // CHECK: VarDecl {{.*}} i2_2 'vector<int, 2>' cinit
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 2>' <HLSLVectorTruncation>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4>' <FloatingToIntegral>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<double, 4>' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'vector<double, 4>' lvalue Var {{.*}} 'd4' 'vector<double, 4>'
  // expected-warning@#i2_2{{implicit conversion truncates vector: 'vector<double, 4>' (vector of 4 'double' values) to 'vector<int, 2>' (vector of 2 'int' values)}}
  // expected-warning@#i2_2{{implicit conversion turns floating-point number into integer: 'vector<double, 4>' (vector of 4 'double' values) to 'vector<int, 2>' (vector of 2 'int' values)}}
  vector<int,2> i2_2 = d4; // #i2_2


  // CHECK: VarDecl {{.*}} used i64_4 'vector<long, 4>' cinit
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<long, 4>' <FloatingToIntegral>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<double, 4>' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'vector<double, 4>' lvalue Var {{.*}} 'd4' 'vector<double, 4>'
  // expected-warning@#i64_4{{implicit conversion turns floating-point number into integer: 'vector<double, 4>' (vector of 4 'double' values) to 'vector<long, 4>' (vector of 4 'long' values)}}
  vector<long,4> i64_4 = d4; // #i64_4

  // CHECK: VarDecl {{.*}} used i2_3 'vector<int, 2>' cinit
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 2>' <HLSLVectorTruncation>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4>' <IntegralCast>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<long, 4>' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'vector<long, 4>' lvalue Var {{.*}} 'i64_4' 'vector<long, 4>'
  // expected-warning@#i2_3{{implicit conversion loses integer precision: 'vector<long, 4>' (vector of 4 'long' values) to 'vector<int, 2>' (vector of 2 'int' values)}}
  // expected-warning@#i2_3{{implicit conversion truncates vector: 'vector<long, 4>' (vector of 4 'long' values) to 'vector<int, 2>' (vector of 2 'int' values)}}
  vector<int,2> i2_3 = i64_4; // #i2_3

  //CHECK: VarDecl {{.*}} b2 'vector<bool, 2>' cinit
  //CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<bool, 2>' <IntegralToBoolean>
  //CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 2>' <LValueToRValue>
  //CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, 2>' lvalue Var {{.*}} 'i2_3' 'vector<int, 2>'
  vector<bool, 2> b2 = i2_3; // No warning for integer to bool conversion.

  // CHECK: VarDecl {{.*}} b2_2 'vector<bool, 2>' cinit
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<bool, 2>' <HLSLVectorTruncation>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<bool, 4>' <FloatingToBoolean>
  // CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<double, 4>' <LValueToRValue>
  // CHECK-NEXT: DeclRefExpr {{.*}} 'vector<double, 4>' lvalue Var {{.*}} 'd4' 'vector<double, 4>'
  // expected-warning@#b2_2{{implicit conversion truncates vector: 'vector<double, 4>' (vector of 4 'double' values) to 'vector<bool, 2>' (vector of 2 'bool' values)}}
  // expected-warning@#b2_2{{implicit conversion turns floating-point number into integer: 'vector<double, 4>' (vector of 4 'double' values) to 'vector<bool, 2>' (vector of 2 'bool' values)}}
  vector<bool, 2> b2_2 = d4; // #b2_2
}

#ifndef NO_ERR

void illegal() {
  // vector extension is illegal
  vector<float,3> f3 = 1.0;
  vector<float,4> f4 = f3; // expected-error{{cannot initialize a variable of type 'vector<[...], 4>' with an lvalue of type 'vector<[...], 3>'}}
}

#endif
