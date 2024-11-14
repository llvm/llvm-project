// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s


// CHECK: -HLSLResourceClassAttr 0x{{[0-9a-f]+}} <col:26> SRV
struct Eg1 {
  [[hlsl::resource_class(SRV)]] int i;  
};

Eg1 e1;

// CHECK: -CXXRecordDecl 0x{{[0-9a-f]+}} <line:13:1, line:15:1> line:13:8 referenced struct Eg2 definition
// CHECK: -HLSLResourceClassAttr 0x{{[0-9a-f]+}} <col:26> UAV
struct Eg2 {
  [[hlsl::resource_class(UAV)]] int i;
};
Eg2 e2;

// CHECK: -CXXRecordDecl 0x{{[0-9a-f]+}} <line:20:1, line:22:1> line:20:8 referenced struct Eg3 definition
// CHECK: -HLSLResourceClassAttr 0x{{[0-9a-f]+}} <col:26> CBuffer
struct Eg3 {
  [[hlsl::resource_class(CBuffer)]] int i;
}; 
Eg3 e3;

// CHECK: -CXXRecordDecl 0x{{[0-9a-f]+}} <line:27:1, line:29:1> line:27:8 referenced struct Eg4 definition
// CHECK: -HLSLResourceClassAttr 0x{{[0-9a-f]+}} <col:26> Sampler
struct Eg4 {
  [[hlsl::resource_class(Sampler)]] int i;
};
Eg4 e4;

RWBuffer<int> In : register(u1);
