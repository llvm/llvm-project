// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-library -finclude-default-header -ast-dump -o - %s | FileCheck %s -check-prefixes=CHECK,SPV
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-library -finclude-default-header -ast-dump -o - %s | FileCheck %s -check-prefixes=CHECK,DXIL

// vk::builtin only exists when targeting SPIR-V.

// CHECK: CXXRecordDecl {{.*}} struct S definition
// CHECK: FieldDecl {{.*}} a 'float'
// SPV-NEXT: HLSLVkBuiltInAttr {{.*}} "PointSize"
// DXIL-NOT: HLSLVkBuiltInAttr
// CHECK: HLSLParsedSemanticAttr {{.*}} "PSIZE"
// CHECK: FieldDecl {{.*}} b 'float'
// CHECK-NEXT: HLSLVkLocationAttr {{.*}} 1
// CHECK-NEXT: HLSLParsedSemanticAttr {{.*}} "B"
struct S {
  [[vk::builtin("PointSize")]] float a : PSIZE;
  [[vk::location(1)]] float b : B;
};

// CHECK: FunctionDecl {{.*}} param 'float (float)'
// CHECK-NEXT: ParmVarDecl {{.*}} p 'float'
// SPV-NEXT: HLSLVkBuiltInAttr {{.*}} "PointSize"
// DXIL-NOT: HLSLVkBuiltInAttr
// CHECK: HLSLParsedSemanticAttr {{.*}} "PSIZE"
float param([[vk::builtin("PointSize")]] float p : PSIZE) : A { return p; }

// CHECK: FunctionDecl {{.*}} func 'float ()'
// SPV: HLSLVkBuiltInAttr {{.*}} "PointSize"
// DXIL-NOT: HLSLVkBuiltInAttr
[[vk::builtin("PointSize")]] float func() : PSIZE { return 1.0f; }
