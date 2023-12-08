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
// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:32> col:10 used Vec2 'float2':'float __attribute__((ext_vector_type(2)))' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:17, col:32> 'float2':'float __attribute__((ext_vector_type(2)))' functional cast to float2 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:24, col:29> 'float2':'float __attribute__((ext_vector_type(2)))'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:24> 'float' <FloatingCast>
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:24> 'double' 1.000000e+00
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:29> 'float' <FloatingCast>
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:29> 'double' 2.000000e+00


// For the float 3 things get fun...
// Here we expect accesses to the vec2 to provide the first and second
// components using ArraySubscriptExpr
// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:33> col:10 Vec3 'float3':'float __attribute__((ext_vector_type(3)))' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:17, col:33> 'float3':'float __attribute__((ext_vector_type(3)))' functional cast to float3 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:24, col:30> 'float3':'float __attribute__((ext_vector_type(3)))'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:24, <invalid sloc>> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:24, <invalid sloc>> 'float' lvalue
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:24> 'float2':'float __attribute__((ext_vector_type(2)))' lvalue Var 0x{{[0-9a-fA-F]+}} 'Vec2' 'float2':'float __attribute__((ext_vector_type(2)))'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> 'int' 0
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:24, <invalid sloc>> 'float' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr 0x{{[0-9a-fA-F]+}} <col:24, <invalid sloc>> 'float' lvalue
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:24> 'float2':'float __attribute__((ext_vector_type(2)))' lvalue Var 0x{{[0-9a-fA-F]+}} 'Vec2' 'float2':'float __attribute__((ext_vector_type(2)))'
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <<invalid sloc>> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:30> 'float' <FloatingCast>
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:30> 'double' 3.000000e+00

// CHECK: VarDecl 0x{{[0-9a-fA-F]+}} <col:3, col:38> col:10 Vec3b 'float3':'float __attribute__((ext_vector_type(3)))' cinit
// CHECK-NEXT: CXXFunctionalCastExpr 0x{{[0-9a-fA-F]+}} <col:18, col:38> 'float3':'float __attribute__((ext_vector_type(3)))' functional cast to float3 <NoOp>
// CHECK-NEXT: InitListExpr 0x{{[0-9a-fA-F]+}} <col:25, col:35> 'float3':'float __attribute__((ext_vector_type(3)))'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:25> 'float' <FloatingCast>
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:25> 'double' 1.000000e+00
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:30> 'float' <FloatingCast>
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:30> 'double' 2.000000e+00
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:35> 'float' <FloatingCast>
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:35> 'double' 3.000000e+00

// The tests above verify pretty explictily that the Initialization lists are
// being constructed as expected. The next tests are bit sparser for brevity.

  float f = 1.0f, g = 2.0f;
  float2 foo0 = float2(f, g); // Non-literal

// CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:54:3, col:29>
// CHECK-NEXT: VarDecl
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:24> 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:24> 'float' lvalue Var  0x{{[0-9a-fA-F]+}} 'f' 'float'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:27> 'float' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:27> 'float' lvalue Var  0x{{[0-9a-fA-F]+}} 'g' 'float'

  int i = 1, j = 2;
  float2 foo1 = float2(1, 2); // Integer literals

// CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:66:3, col:29>
// CHECK-NEXT: VarDecl
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:24> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:24> 'int' 1
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:27> 'float' <IntegralToFloating>
// CHECK-NEXT: IntegerLiteral 0x{{[0-9a-fA-F]+}} <col:27> 'int' 2

  float2 foo2 = float2(i, j); // Integer non-literal

// CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:77:3, col:29>
// CHECK-NEXT: VarDecl
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:24> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:24> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:24> 'int' lvalue Var 0x{{[0-9a-fA-F]+}} 'i' 'int'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:27> 'float' <IntegralToFloating>
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:27> 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:27> 'int' lvalue Var 0x{{[0-9a-fA-F]+}} 'j' 'int'

  struct S { float f; } s;
  float2 foo4 = float2(s.f, s.f);

// CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:91:3, col:33>
// CHECK-NEXT: VarDecl
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:24, col:26> 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:24, col:26> 'float' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:24> 'struct S':'S' lvalue Var 0x{{[0-9a-fA-F]+}} 's' 'struct S':'S'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:29, col:31> 'float' <LValueToRValue>
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:29, col:31> 'float' lvalue .f 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:29> 'struct S':'S' lvalue Var 0x{{[0-9a-fA-F]+}} 's' 'struct S':'S'

  struct T {
    operator float() const { return 1.0f; }
  } t;
  float2 foo5 = float2(t, t); // user-defined cast operator

// CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:107:3, col:29>
// CHECK-NEXT: VarDecl
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:24> 'float' <UserDefinedConversion>
// CHECK-NEXT: CXXMemberCallExpr 0x{{[0-9a-fA-F]+}} <col:24> 'float'
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:24> '<bound member function type>' .operator float 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:24> 'const T' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:24> 'struct T':'T' lvalue Var 0x{{[0-9a-fA-F]+}} 't' 'struct T':'T'
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:27> 'float' <UserDefinedConversion>
// CHECK-NEXT: CXXMemberCallExpr 0x{{[0-9a-fA-F]+}} <col:27> 'float'
// CHECK-NEXT: MemberExpr 0x{{[0-9a-fA-F]+}} <col:27> '<bound member function type>' .operator float 0x{{[0-9a-fA-F]+}}
// CHECK-NEXT: ImplicitCastExpr 0x{{[0-9a-fA-F]+}} <col:27> 'const T' lvalue <NoOp>
// CHECK-NEXT: DeclRefExpr 0x{{[0-9a-fA-F]+}} <col:27> 'struct T':'T' lvalue Var 0x{{[0-9a-fA-F]+}} 't' 'struct T':'T'

  typedef float2 second_level_of_typedefs;
  second_level_of_typedefs foo6 = float2(1.0f, 2.0f);

// CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:125:3, col:53>
// CHECK-NEXT: VarDecl
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:42> 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:48> 'float' 2.000000e+00

  float2 foo7 = second_level_of_typedefs(1.0f, 2.0f);

// CHECK: DeclStmt 0x{{[0-9a-fA-F]+}} <line:134:3, col:53>
// CHECK-NEXT: VarDecl
// CHECK-NEXT: CXXFunctionalCastExpr
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:42> 'float' 1.000000e+00
// CHECK-NEXT: FloatingLiteral 0x{{[0-9a-fA-F]+}} <col:48> 'float' 2.000000e+00

}
