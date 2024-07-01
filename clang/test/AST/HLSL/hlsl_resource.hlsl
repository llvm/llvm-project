// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -syntax-only -disable-llvm-passes -o - %s | FileCheck %s

struct MyBuffer {
  __builtin_hlsl_resource_t handle;
};

// CHECK:TypedefDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> implicit referenced __builtin_hlsl_resource_t '__builtin_hlsl_resource_t'
// CHECK-NEXT:BuiltinType 0x{{[0-9a-f]+}} '__builtin_hlsl_resource_t'

// CHECK:CXXRecordDecl 0x{{[0-9a-f]+}} <clang/test/AST/HLSL/hlsl_resource.hlsl:3:1, line:5:1> line:3:8 struct MyBuffer definition
// CHECK:FieldDecl 0x{{[0-9a-f]+}} <line:4:3, col:29> col:29 handle '__builtin_hlsl_resource_t'
