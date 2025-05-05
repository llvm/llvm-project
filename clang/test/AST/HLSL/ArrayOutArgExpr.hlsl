// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump %s | FileCheck %s

// CHECK-LABEL: increment
void increment(inout int Arr[2]) {
  for (int I = 0; I < 2; I++)
    Arr[0] += 2;
}

// CHECK-LABEL: call
// CHECK: CallExpr 0x{{.*}} {{.*}} 'void'
// CHECK: ImplicitCastExpr 0x{{.*}} {{.*}} 'void (*)(inout int[2])' <FunctionToPointerDecay>
// CHECK: DeclRefExpr 0x{{.*}} {{.*}} 'void (inout int[2])' lvalue Function 0x{{.*}} 'increment' 'void (inout int[2])'
// CHECK: HLSLOutArgExpr 0x{{.*}} {{.*}} 'int[2]' lvalue inout
// CHECK: OpaqueValueExpr [[A:0x.*]] {{.*}} 'int[2]' lvalue
// CHECK: DeclRefExpr [[B:0x.*]] {{.*}} 'int[2]' lvalue Var [[E:0x.*]] 'A' 'int[2]'
// CHECK: OpaqueValueExpr [[C:0x.*]] {{.*}} 'int[2]' lvalue
// CHECK: ImplicitCastExpr [[D:0x.*]] {{.*}} 'int[2]' <HLSLArrayRValue>
// CHECK: OpaqueValueExpr [[A]] {{.*}} 'int[2]' lvalue
// CHECK: DeclRefExpr [[B]] {{.*}} 'int[2]' lvalue Var [[E]] 'A' 'int[2]'
// CHECK: BinaryOperator 0x{{.*}} {{.*}} 'int[2]' lvalue '='
// CHECK: OpaqueValueExpr [[A]] {{.*}} 'int[2]' lvalue
// CHECK: DeclRefExpr 0x{{.*}} {{.*}} 'int[2]' lvalue Var [[E]] 'A' 'int[2]'
// CHECK: ImplicitCastExpr 0x{{.*}} {{.*}} 'int[2]' <HLSLArrayRValue>
// CHECK: OpaqueValueExpr [[C]] {{.*}} 'int[2]' lvalue
// CHECK: ImplicitCastExpr [[D]] {{.*}} 'int[2]' <HLSLArrayRValue>
// CHECK: OpaqueValueExpr [[A]] {{.*}} 'int[2]' lvalue
// CHECK: DeclRefExpr [[B]] {{.*}} 'int[2]' lvalue Var [[E]] 'A' 'int[2]'
export int call() {
  int A[2] = { 0, 1 };
  increment(A);
  return A[0];
}

// CHECK-LABEL: fn2
void fn2(out int Arr[2]) {
  Arr[0] += 5;
  Arr[1] += 6;
}

// CHECK-LABEL: call2
// CHECK: CallExpr 0x{{.*}} {{.*}} 'void'
// CHECK: ImplicitCastExpr 0x{{.*}} {{.*}} 'void (*)(out int[2])' <FunctionToPointerDecay>
// CHECK: DeclRefExpr 0x{{.*}} {{.*}} 'void (out int[2])' lvalue Function 0x{{.*}} 'fn2' 'void (out int[2])'
// CHECK: HLSLOutArgExpr 0x{{.*}} {{.*}} 'int[2]' lvalue out
// CHECK: OpaqueValueExpr [[A:0x.*]] {{.*}} 'int[2]' lvalue
// CHECK: DeclRefExpr [[B:0x.*]] {{.*}} 'int[2]' lvalue Var [[E:0x.*]] 'A' 'int[2]'
// CHECK: OpaqueValueExpr [[C:0x.*]] {{.*}} 'int[2]' lvalue
// CHECK: ImplicitCastExpr [[D:0x.*]] {{.*}} 'int[2]' <HLSLArrayRValue>
// CHECK: OpaqueValueExpr [[A]] {{.*}} 'int[2]' lvalue
// CHECK: DeclRefExpr [[B]] {{.*}} 'int[2]' lvalue Var [[E]] 'A' 'int[2]'
// CHECK: BinaryOperator 0x{{.*}} {{.*}} 'int[2]' lvalue '='
// CHECK: OpaqueValueExpr [[A]] {{.*}} 'int[2]' lvalue
// CHECK: DeclRefExpr [[B]] {{.*}} 'int[2]' lvalue Var [[E]] 'A' 'int[2]'
// CHECK: ImplicitCastExpr 0x{{.*}} {{.*}} 'int[2]' <HLSLArrayRValue>
// CHECK: OpaqueValueExpr [[C]] {{.*}} 'int[2]' lvalue
// CHECK: ImplicitCastExpr [[D]] {{.*}} 'int[2]' <HLSLArrayRValue>
// CHECK: OpaqueValueExpr [[A]] {{.*}} 'int[2]' lvalue
// CHECK: DeclRefExpr [[B]] {{.*}} 'int[2]' lvalue Var [[E]] 'A' 'int[2]'
export int call2() {
  int A[2] = { 0, 1 };
  fn2(A);
  return 1;
}
