// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.6-library -x hlsl -ast-dump -disable-llvm-passes -o - 2>&1 %s | FileCheck %s

struct MyBuffer {
  int8_t4_packed val;
};

// CHECK:CXXRecordDecl 0x{{[0-9a-z]+}} <{{.*}}> line:3:8 struct MyBuffer definition
// CHECK:FieldDecl 0x{{[0-9a-z]+}} <line:4:3, col:18> col:18 val 'int8_t4_packed'
