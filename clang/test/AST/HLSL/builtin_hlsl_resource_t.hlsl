// RUN: not %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -disable-llvm-passes -o - 2>&1 %s | FileCheck %s

struct MyBuffer {
  __builtin_hlsl_resource_t handle;
};

// CHECK-DAG: error: field has sizeless type '__builtin_hlsl_resource_t'

// CHECK:CXXRecordDecl 0x{{[0-9a-f]+}} <{{.*}}> line:3:8 invalid struct MyBuffer definition
// CHECK:FieldDecl 0x{{[0-9a-f]+}} <line:4:3, col:29> col:29 invalid handle '__builtin_hlsl_resource_t'
