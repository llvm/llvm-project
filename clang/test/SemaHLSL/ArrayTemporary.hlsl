// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ast-dump %s | FileCheck %s

void fn(float x[2]) { }

// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float[2])' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (float[2])' lvalue Function {{.*}} 'fn' 'void (float[2])'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float[2]' <HLSLArrayRValue>

void call() {
  float Arr[2] = {0, 0};
  fn(Arr);
}

struct Obj {
  float V;
  int X;
};

void fn2(Obj O[4]) { }

// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(Obj[4])' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (Obj[4])' lvalue Function {{.*}} 'fn2' 'void (Obj[4])'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'Obj[4]' <HLSLArrayRValue>

void call2() {
  Obj Arr[4] = {};
  fn2(Arr);
}


void fn3(float x[2][2]) { }

// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float[2][2])' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (float[2][2])' lvalue Function {{.*}} 'fn3' 'void (float[2][2])'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float[2][2]' <HLSLArrayRValue>

void call3() {
  float Arr[2][2] = {{0, 0}, {1,1}};
  fn3(Arr);
}

// This template function should be instantiated 3 times for the different array
// types and lengths.

// CHECK: FunctionTemplateDecl {{.*}} template_fn
// CHECK-NEXT: TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
// CHECK-NEXT: FunctionDecl {{.*}} template_fn 'void (T)'
// CHECK-NEXT: ParmVarDecl {{.*}} Val 'T'

// CHECK: FunctionDecl {{.*}} used template_fn 'void (float[2])' implicit_instantiation
// CHECK-NEXT: TemplateArgument type 'float[2]'
// CHECK-NEXT: ArrayParameterType {{.*}} 'float[2]' 2
// CHECK-NEXT: BuiltinType {{.*}} 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} Val 'float[2]'

// CHECK: FunctionDecl {{.*}} used template_fn 'void (float[4])' implicit_instantiation
// CHECK-NEXT: TemplateArgument type 'float[4]'
// CHECK-NEXT: ArrayParameterType {{.*}} 'float[4]' 4
// CHECK-NEXT: BuiltinType {{.*}} 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} Val 'float[4]'

// CHECK: FunctionDecl {{.*}} used template_fn 'void (int[3])' implicit_instantiation
// CHECK-NEXT: TemplateArgument type 'int[3]'
// CHECK-NEXT: ArrayParameterType {{.*}} 'int[3]' 3
// CHECK-NEXT: BuiltinType {{.*}} 'int'
// CHECK-NEXT: ParmVarDecl {{.*}} Val 'int[3]'

template<typename T>
void template_fn(T Val) {}

// CHECK: FunctionDecl {{.*}} call 'void (float[2], float[4], int[3])'
// CHECK: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float[2])' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (float[2])' lvalue Function {{.*}} 'template_fn' 'void (float[2])' (FunctionTemplate {{.*}} 'template_fn')
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float[2]' <HLSLArrayRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float[2]' lvalue ParmVar {{.*}} 'FA2' 'float[2]'
// CHECK-NEXT: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(float[4])' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (float[4])' lvalue Function {{.*}} 'template_fn' 'void (float[4])' (FunctionTemplate {{.*}} 'template_fn')
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float[4]' <HLSLArrayRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'float[4]' lvalue ParmVar {{.*}} 'FA4' 'float[4]'
// CHECK-NEXT: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(int[3])' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (int[3])' lvalue Function {{.*}} 'template_fn' 'void (int[3])' (FunctionTemplate {{.*}} 'template_fn')
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int[3]' <HLSLArrayRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int[3]' lvalue ParmVar {{.*}} 'IA3' 'int[3]'

void call(float FA2[2], float FA4[4], int IA3[3]) {
  template_fn(FA2);
  template_fn(FA4);
  template_fn(IA3);
}
