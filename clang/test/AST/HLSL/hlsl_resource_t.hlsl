// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -disable-llvm-passes -o - 2>&1 %s | FileCheck %s

struct MyBuffer {
  __hlsl_resource_t handle;
};

// CHECK:CXXRecordDecl 0x{{[0-9a-f]+}} <{{.*}}> line:3:8 struct MyBuffer definition
// CHECK:FieldDecl 0x{{[0-9a-f]+}} <line:4:3, col:21> col:21 handle '__hlsl_resource_t'
