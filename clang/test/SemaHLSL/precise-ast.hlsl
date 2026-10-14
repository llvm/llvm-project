// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -ast-dump %s | FileCheck %s

struct S {
  precise float F;
};
// CHECK-LABEL: CXXRecordDecl {{.*}} struct S definition
// CHECK: FieldDecl {{.*}} F 'float'
// CHECK-NEXT: HLSLPreciseAttr

precise float Fn(precise float P) {
  precise float L = P;
  float precise M = L;
  return M;
}
// CHECK-LABEL: FunctionDecl {{.*}} Fn 'float (float)'
// CHECK: ParmVarDecl {{.*}} used P 'float'
// CHECK-NEXT: HLSLPreciseAttr
// CHECK: VarDecl {{.*}} used L 'float' cinit
// CHECK: HLSLPreciseAttr
// CHECK: VarDecl {{.*}} used M 'float' cinit
// CHECK: HLSLPreciseAttr
// CHECK: HLSLPreciseAttr

template <typename T> T Tmpl(T V) {
  precise T X = V;
  return X;
}
float Inst() { return Tmpl<float>(1.0); }
// CHECK-LABEL: FunctionTemplateDecl {{.*}} Tmpl
// CHECK: VarDecl {{.*}} X 'T' nrvo cinit
// CHECK: HLSLPreciseAttr
// CHECK: FunctionDecl {{.*}} used Tmpl 'float (float)' implicit_instantiation
// CHECK: VarDecl {{.*}} X 'float' cinit
// CHECK: HLSLPreciseAttr
