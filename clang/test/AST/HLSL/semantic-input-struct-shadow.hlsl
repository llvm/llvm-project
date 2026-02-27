// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -finclude-default-header -ast-dump -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-vertex -finclude-default-header -ast-dump -o - %s | FileCheck %s


// CHECK: CXXRecordDecl {{.*}} referenced struct S definition
// CHECK: FieldDecl {{.*}} field1 'int'
// CHECK-NEXT: HLSLParsedSemanticAttr {{.*}} "A" 0
// CHECK: FieldDecl {{.*}} field2 'int'
// CHECK-NEXT: HLSLParsedSemanticAttr {{.*}} "B" 4

struct S {
  int field1 : A;
  int field2 : B4;
};

// CHECK:       FunctionDecl {{.*}} main 'void (S)'
// CHECK-NEXT:  ParmVarDecl {{.*}} s 'S'
// CHECK-NEXT:  HLSLParsedSemanticAttr {{.*}} "C" 0
// CHECK-NEXT:  HLSLAppliedSemanticAttr {{.*}} "C" 0
// CHECK-NEXT:  HLSLAppliedSemanticAttr {{.*}} "C" 1
void main(S s : C) {}
