// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s

typedef float float2 __attribute__((ext_vector_type(2)));
typedef float float3 __attribute__((ext_vector_type(3)));

[numthreads(1,1,1)]
void entry() {
  float2 Vec2 = float2(1.0, 2.0);
  float3 Vec3 = float3(Vec2, 3.0);
  float3 Vec3b = float3(1.0, 2.0, 3.0);

// For the float2 vector, we just expect a conversion from constructor
// parameters to an initialization list
// CHECK-LABEL: VarDecl 0x{{[0-9a-fA-F]+}} {{.*}} used Vec2 'float2':'vector<float, 2>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float2':'vector<float, 2>' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float2':'vector<float, 2>'
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} {{.*}} 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} {{.*}} 'float' 2.000000e+00


// For the float 3 things get fun...
// Here we expect accesses to the vec2 to provide the first and second
// components using ArraySubscriptExpr
// CHECK-LABEL: VarDecl 0x{{[0-9a-fA-F]+}} {{.*}} col:10 Vec3 'float3':'vector<float, 3>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float3':'vector<float, 3>' functional cast to float3 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float3':'vector<float, 3>'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:24, <invalid sloc>> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:24, <invalid sloc>> 'float' lvalue
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float2':'vector<float, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'Vec2' 'float2':'vector<float, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:24, <invalid sloc>> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:24, <invalid sloc>> 'float' lvalue
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float2':'vector<float, 2>' lvalue Var 0x{{[0-9a-fA-F]+}} 'Vec2' 'float2':'vector<float, 2>'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} {{.*}} 'float' 3.000000e+00

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} {{.*}} col:10 Vec3b 'float3':'vector<float, 3>' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float3':'vector<float, 3>' functional cast to float3 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float3':'vector<float, 3>'

// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} {{.*}} 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} {{.*}} 'float' 2.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} {{.*}} 'float' 3.000000e+00

// The tests above verify pretty explictily that the Initialization lists are
// being constructed as expected. The next tests are bit sparser for brevity.

  float f = 1.0f, g = 2.0f;
  float2 foo0 = float2(f, g); // Non-literal

// CHECK-LABEL: VarDecl 0x{{[0-9a-fA-F]+}} {{.*}} foo0 'float2'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' lvalue Var  0x{{[0-9a-fA-F]+}} 'f' 'float'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' lvalue Var  0x{{[0-9a-fA-F]+}} 'g' 'float'

  int i = 1, j = 2;
  float2 foo1 = float2(1, 2); // Integer literals

// CHECK-LABEL: VarDecl 0x{{[0-9a-fA-F]+}} {{.*}} foo1 'float2'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} {{.*}} 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} {{.*}} 'int' 2

  float2 foo2 = float2(i, j); // Integer non-literal

// CHECK-LABEL: VarDecl 0x{{[0-9a-fA-F]+}} {{.*}} foo2 'float2'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'int' lvalue Var 0x{{[0-9a-fA-F]+}} 'i' 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'int' lvalue Var 0x{{[0-9a-fA-F]+}} 'j' 'int'

  struct S { float f; } s;
  float2 foo4 = float2(s.f, s.f);

// CHECK-LABEL: VarDecl 0x{{[0-9a-fA-F]+}} {{.*}} foo4 'float2'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'struct S':'S' lvalue Var 0x{{[0-9a-fA-F]+}} 's' 'struct S':'S'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'struct S':'S' lvalue Var 0x{{[0-9a-fA-F]+}} 's' 'struct S':'S'

  struct T {
    operator float() const { return 1.0f; }
  } t;
  float2 foo5 = float2(t, t); // user-defined cast operator

// CHECK-LABEL: VarDecl 0x{{[0-9a-fA-F]+}} {{.*}} foo5 'float2'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' <UserDefinedConversion>
// CHECK-NEXT: CXXMemberCallExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float'
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} {{.*}} '<bound member function type>' .operator float 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'const T' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'struct T':'T' lvalue Var 0x{{[0-9a-fA-F]+}} 't' 'struct T':'T'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float' <UserDefinedConversion>
// CHECK-NEXT: CXXMemberCallExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'float'
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} {{.*}} '<bound member function type>' .operator float 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'const T' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} {{.*}} 'struct T':'T' lvalue Var 0x{{[0-9a-fA-F]+}} 't' 'struct T':'T'

  typedef float2 second_level_of_typedefs;
  second_level_of_typedefs foo6 = float2(1.0f, 2.0f);


// CHECK-LABEL: VarDecl 0x{{[0-9a-fA-F]+}} {{.*}} foo6 'second_level_of_typedefs'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} {{.*}} 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} {{.*}} 'float' 2.000000e+00

  float2 foo7 = second_level_of_typedefs(1.0f, 2.0f);

// CHECK-LABEL: VarDecl 0x{{[0-9a-fA-F]+}} {{.*}} foo7 'float2'
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} {{.*}} 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} {{.*}} 'float' 2.000000e+00

}
