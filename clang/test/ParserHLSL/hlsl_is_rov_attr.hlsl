// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s


// CHECK: -HLSLROVAttr 0x{{[0-9a-f]+}} <col:10, col:16>
struct [[hlsl::is_rov]] Eg1 {
  int i;  
};

Eg1 e1;
