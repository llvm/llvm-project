// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -ast-dump %s | FileCheck %s

struct MyStruct {
  float f;
  RWBuffer<float> Buf;

  void Store() const {
    Buf[0] = f;
  }
};

cbuffer CB {
  MyStruct one;
}

MyStruct two;

// CHECK: FunctionDecl {{.*}} main 'void ()'
[numthreads(1, 1, 1)]
void main() {
// CHECK: ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .Store
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'const MyStruct' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const MyStruct' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl_constant MyStruct' lvalue Var {{.*}} 'one' 'hlsl_constant MyStruct'
  one.Store();

// CHECK: ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} '<bound member function type>' .Store {{.*}}
// CHECK-NEXT: MaterializeTemporaryExpr {{.*}} 'const MyStruct' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const MyStruct' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl_constant MyStruct' lvalue Var {{.*}} 'two' 'hlsl_constant MyStruct'
  two.Store();
}
