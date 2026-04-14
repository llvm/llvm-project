// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-compute -x hlsl -ast-dump -o - %s | FileCheck %s

// Verify that HLSL vector/matrix paren-syntax constructors inside template
// functions produce the correct AST after template instantiation. The key
// property is that the CXXFunctionalCastExpr wrapping an InitListExpr survives
// re-instantiation intact.

typedef int int2 __attribute__((ext_vector_type(2)));
typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));
typedef float float4 __attribute__((ext_vector_type(4)));
typedef float float2x2 __attribute__((matrix_type(2, 2)));

// Basic vector constructors in a template.
template<typename T>
void vector_constructors() {
// CHECK-LABEL: FunctionDecl {{.*}} used vector_constructors 'void ()' implicit_instantiation
// CHECK: VarDecl {{.*}} a 'int2':'vector<int, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr {{.*}} 'int2':'vector<int, 2>' functional cast to int2 <NoOp>
// CHECK-NEXT: InitListExpr {{.*}} 'int2':'vector<int, 2>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
// CHECK: VarDecl {{.*}} b 'float3':'vector<float, 3>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr {{.*}} 'float3':'vector<float, 3>' functional cast to float3 <NoOp>
// CHECK-NEXT: InitListExpr {{.*}} 'float3':'vector<float, 3>'
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 2.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 3.000000e+00
  int2 a = int2(0, 0);
  float3 b = float3(1.0f, 2.0f, 3.0f);
}


// Nested vector constructor: float4 built from a float2 + two scalars.
template<typename T>
void vector_constructors_nested() {
// CHECK-LABEL: FunctionDecl {{.*}} used vector_constructors_nested 'void ()' implicit_instantiation
// CHECK: VarDecl {{.*}} used v 'float2':'vector<float, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr {{.*}} 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr {{.*}} 'float2':'vector<float, 2>'
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 2.000000e+00
// CHECK: VarDecl {{.*}} w 'float4':'vector<float, 4>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr {{.*}} 'float4':'vector<float, 4>' functional cast to float4 <NoOp>
// CHECK-NEXT: InitListExpr {{.*}} 'float4':'vector<float, 4>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2':'vector<float, 2>' lvalue Var {{.*}} 'v' 'float2':'vector<float, 2>'
// CHECK-NEXT: IntegerLiteral {{.*}} '__size_t':'unsigned long' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue vectorcomponent
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2':'vector<float, 2>' lvalue Var {{.*}} 'v' 'float2':'vector<float, 2>'
// CHECK-NEXT: IntegerLiteral {{.*}} '__size_t':'unsigned long' 1
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 3.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 4.000000e+00
  float2 v = float2(1.0f, 2.0f);
  float4 w = float4(v, 3.0f, 4.0f);
}

// Matrix constructor in a template.
template<typename T>
void matrix_constructors() {
// CHECK-LABEL: FunctionDecl {{.*}} used matrix_constructors 'void ()' implicit_instantiation
// CHECK: VarDecl {{.*}} m 'float2x2':'matrix<float, 2, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr {{.*}} 'float2x2':'matrix<float, 2, 2>' functional cast to float2x2 <NoOp>
// CHECK-NEXT: InitListExpr {{.*}} 'float2x2':'matrix<float, 2, 2>'
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 2.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 3.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 4.000000e+00
  float2x2 m = float2x2(1.0f, 2.0f, 3.0f, 4.0f);
}

// Instantiate the templates from a compute shader entry point.
[numthreads(1,1,1)]
void main() {
  vector_constructors<int>();
  vector_constructors_nested<float>();
  matrix_constructors<int>();
}
