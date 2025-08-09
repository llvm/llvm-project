// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -ast-dump -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -std=hlsl202x -ast-dump -o - %s | FileCheck %s

typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));

[numthreads(1,1,1)]
void entry() {
  float2 Vec2 = float2(1.0, 2.0);
  float3 Vec3 = float3(Vec2, 3.0);
  float3 Vec3b = float3(1.0, 2.0, 3.0);

// For the float2 vector, we just expect a conversion from constructor
// parameters to an initialization list
// CHECK-LABEL: VarDecl {{.*}} used Vec2 'float2':'vector<float, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr {{.*}} 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr {{.*}} 'float2':'vector<float, 2>'
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 2.000000e+00


// For the float 3 things get fun...
// Here we expect accesses to the vec2 to provide the first and second
// components using ArraySubscriptExpr
// CHECK-LABEL: VarDecl {{.*}} Vec3 'float3':'vector<float, 3>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr {{.*}} 'float3':'vector<float, 3>' functional cast to float3 <NoOp>
// CHECK-NEXT: InitListExpr {{.*}} 'float3':'vector<float, 3>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2':'vector<float, 2>' lvalue Var {{.*}} 'Vec2' 'float2':'vector<float, 2>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 0
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'float' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'float2':'vector<float, 2>' lvalue Var {{.*}} 'Vec2' 'float2':'vector<float, 2>'
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 3.000000e+00

// CHECK: VarDecl {{.*}} 'float3':'vector<float, 3>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr {{.*}} 'float3':'vector<float, 3>' functional cast to float3 <NoOp>
// CHECK-NEXT: InitListExpr {{.*}} 'float3':'vector<float, 3>'

// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 2.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 3.000000e+00

// The tests above verify pretty explictily that the Initialization lists are
// being constructed as expected. The next tests are bit sparser for brevity.

  float f = 1.0f, g = 2.0f;
  float2 foo0 = float2(f, g); // Non-literal

// CHECK-LABEL: VarDecl {{.*}} foo0 'float2'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue Var  {{.*}} 'f' 'float'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue Var  {{.*}} 'g' 'float'

  int i = 1, j = 2;
  float2 foo1 = float2(1, 2); // Integer literals

// CHECK-LABEL: VarDecl {{.*}} foo1 'float2'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 1
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral {{.*}} 'int' 2

  float2 foo2 = float2(i, j); // Integer non-literal

// CHECK-LABEL: VarDecl {{.*}} foo2 'float2'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'i' 'int'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue Var {{.*}} 'j' 'int'

  struct S { float f; } s;
  float2 foo4 = float2(s.f, s.f);

// CHECK-LABEL: VarDecl {{.*}} foo4 'float2'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .f {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'struct S' lvalue Var {{.*}} 's' 'struct S'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr {{.*}} 'float' lvalue .f {{.*}}
// CHECK-NEXT: DeclRefExpr {{.*}} 'struct S' lvalue Var {{.*}} 's' 'struct S'

  struct T {
    operator float() const { return 1.0f; }
  } t;
  float2 foo5 = float2(t, t); // user-defined cast operator

// CHECK-LABEL: VarDecl {{.*}} foo5 'float2'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <UserDefinedConversion>
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'float'
// CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .operator float {{.*}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const T' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'struct T' lvalue Var {{.*}} 't' 'struct T'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float' <UserDefinedConversion>
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'float'
// CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .operator float {{.*}}
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const T' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr {{.*}} 'struct T' lvalue Var {{.*}} 't' 'struct T'

  typedef float2 second_level_of_typedefs;
  second_level_of_typedefs foo6 = float2(1.0f, 2.0f);


// CHECK-LABEL: VarDecl {{.*}} foo6 'second_level_of_typedefs'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 2.000000e+00

  float2 foo7 = second_level_of_typedefs(1.0f, 2.0f);

// CHECK-LABEL: VarDecl {{.*}} foo7 'float2'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral {{.*}} 'float' 2.000000e+00

}
