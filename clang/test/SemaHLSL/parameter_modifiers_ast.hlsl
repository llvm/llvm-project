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

// CHECK: FunctionDecl {{.*}} fn3 'void (float &)'
// CHECK-NEXT: ParmVarDecl {{.*}} f 'float &'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} out
// CHECK-NOT: HLSLParamModifierAttr
void fn3(out float f);

// CHECK: FunctionDecl {{.*}} fn4 'void (float &)'
// CHECK-NEXT: ParmVarDecl {{.*}} f 'float &'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout
// CHECK-NOT: HLSLParamModifierAttr
void fn4(inout float f);

// CHECK: FunctionDecl {{.*}} fn5 'void (float &)'
// CHECK-NEXT: ParmVarDecl {{.*}} f 'float &'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout MergedSpelling
// CHECK-NOT: HLSLParamModifierAttr
void fn5(out in float f);

// CHECK: FunctionDecl {{.*}} fn6 'void (float &)'
// CHECK-NEXT: ParmVarDecl {{.*}} f 'float &'
// CHECK-NEXT: HLSLParamModifierAttr {{.*}} inout MergedSpelling
// CHECK-NOT: HLSLParamModifierAttr
void fn6(in out float f);
