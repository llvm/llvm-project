// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library %s -ast-dump | FileCheck %s

// CHECK: FunctionDecl {{.*}} fn 'void (float)'
// CHECK-NEXT: ParmVarDecl {{.*}} f 'float'
// CHECK-NOT: HLSLParamModifierAttr
void fn(float f);

// CHECK: FunctionDecl {{.*}}6 fn2 'void (float)'
// CHECK-NEXT: ParmVarDecl {{.*}} f 'float'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} in
// CHECK-NOT: HLSLParamModifierAttr
void fn2(in float f);

// CHECK: FunctionDecl {{.*}} fn3 'void (out float)'
// CHECK-NEXT: ParmVarDecl {{.*}} f 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NOT: HLSLParamModifierAttr
void fn3(out float f);

// CHECK: FunctionDecl {{.*}} fn4 'void (inout float)'
// CHECK-NEXT: ParmVarDecl {{.*}} f 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout
// CHECK-NOT: HLSLParamModifierAttr
void fn4(inout float f);

// CHECK: FunctionDecl {{.*}} fn5 'void (inout float)'
// CHECK-NEXT: ParmVarDecl {{.*}} f 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout MergedSpelling
// CHECK-NOT: HLSLParamModifierAttr
void fn5(out in float f);

// CHECK: FunctionDecl {{.*}} fn6 'void (inout float)'
// CHECK-NEXT: ParmVarDecl {{.*}} f 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout MergedSpelling
// CHECK-NOT: HLSLParamModifierAttr
void fn6(in out float f);

// CHECK-NEXT: FunctionTemplateDecl [[Template:0x[0-9a-fA-F]+]] {{.*}} fn7
// CHECK-NEXT: TemplateTypeParmDecl {{.*}} referenced typename depth 0 index 0 T
// CHECK-NEXT: FunctionDecl {{.*}} fn7 'void (inout T)'
// CHECK-NEXT: ParmVarDecl {{.*}} f 'T'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout
// CHECK-NEXT: FunctionDecl [[Instantiation:0x[0-9a-fA-F]+]] {{.*}} used fn7 'void (inout float)' implicit_instantiation
// CHECK-NEXT: TemplateArgument type 'float'
// CHECK-NEXT:  BuiltinType {{.*}} 'float'
// CHECK-NEXT: ParmVarDecl {{.*}} f 'float &__restrict'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout

template <typename T>
void fn7(inout T f);

// CHECK: FunctionDecl {{.*}} fn8 'void ()'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} used f 'float'
// CHECK-NEXT: CallExpr {{.*}} 'void'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(inout float)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (inout float)' lvalue
// CHECK-SAME: Function [[Instantiation]] 'fn7' 'void (inout float)'
// CHECK-SAME: (FunctionTemplate [[Template]] 'fn7')
// CHECK-NEXT: HLSLOutArgExpr {{.*}}'float' lvalue
void fn8() {
  float f;
  fn7<float>(f);
}
