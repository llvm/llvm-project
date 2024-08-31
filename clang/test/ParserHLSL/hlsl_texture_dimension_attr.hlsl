// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s


// CHECK: -HLSLTextureDimensionAttr 0x{{[0-9a-f]+}} <col:34> 1
struct [[hlsl::texture_dimension(1)]] Eg1 {
  int i;  
};

Eg1 e1;
