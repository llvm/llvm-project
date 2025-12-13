// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ast-dump %s | FileCheck %s
// Structs are packed by default in HLSL

#include <hlsl/hlsl_basic_types.h>

// CHECK: CXXRecordDecl {{.*}} struct S definition
// CHECK:      DefinitionData
// CHECK:      PackedAttr {{.*}} Implicit
// CHECK-NEXT: CXXRecordDecl {{.*}} implicit struct S
struct S {
  float2 f;
  int i;
};
