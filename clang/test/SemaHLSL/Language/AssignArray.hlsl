// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library %s -ast-dump | FileCheck %s

typedef vector<int,4> int8[2];

export void fn(int8 A) {
  int8 a = {A};
// CHECK-LABEL: VarDecl {{.*}} b 'int8':'vector<int, 4>[2]' cinit
// CHECK-NEXT: ArrayInitLoopExpr {{.*}} 'int8':'vector<int, 4>[2]'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int8':'vector<int, 4>[2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'int8':'vector<int, 4>[2]' lvalue Var {{.*}} 'a' 'int8':'vector<int, 4>[2]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'vector<int, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4> *' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'int8':'vector<int, 4>[2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'int8':'vector<int, 4>[2]' lvalue Var {{.*}} 'a' 'int8':'vector<int, 4>[2]'
// CHECK-NEXT: ArrayInitIndexExpr {{.*}} 'unsigned long'
  int8 b = a;

// CHECK-LABEL: VarDecl {{.*}} c 'int8':'vector<int, 4>[2]' cinit
// CHECK-NEXT: ArrayInitLoopExpr {{.*}} 'int8':'vector<int, 4>[2]'
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'vector<int, 4>[2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, 4>[2]' lvalue ParmVar {{.*}} 'A' 'vector<int, 4>[2]'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4>' <LValueToRValue>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'vector<int, 4>' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'vector<int, 4> *' <ArrayToPointerDecay>
// CHECK-NEXT: OpaqueValueExpr {{.*}} 'vector<int, 4>[2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} 'vector<int, 4>[2]' lvalue ParmVar {{.*}} 'A' 'vector<int, 4>[2]'
// CHECK-NEXT: ArrayInitIndexExpr {{.*}} 'unsigned long'
  int8 c = A;
}




