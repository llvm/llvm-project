// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s


// CHECK: -HLSLResourceClassAttr 0x{{[0-9a-f]+}} <col:31> SRV
struct [[hlsl::resource_class(SRV)]] Eg1 {
  int i;  
};

Eg1 e1;

// CHECK: -CXXRecordDecl 0x{{[0-9a-f]+}} <line:13:1, line:15:1> line:13:38 referenced struct Eg2 definition
// CHECK: -HLSLResourceClassAttr 0x{{[0-9a-f]+}} <col:31> UAV
struct [[hlsl::resource_class(UAV)]] Eg2 {
  int i;
};
Eg2 e2;

// CHECK: -CXXRecordDecl 0x{{[0-9a-f]+}} <line:20:1, line:22:1> line:20:42 referenced struct Eg3 definition
// CHECK: -HLSLResourceClassAttr 0x{{[0-9a-f]+}} <col:31> CBuffer
struct [[hlsl::resource_class(CBuffer)]] Eg3 {
  int i;
}; 
Eg3 e3;

// CHECK: -CXXRecordDecl 0x{{[0-9a-f]+}} <line:27:1, line:29:1> line:27:42 referenced struct Eg4 definition
// CHECK: -HLSLResourceClassAttr 0x{{[0-9a-f]+}} <col:31> Sampler
struct [[hlsl::resource_class(Sampler)]] Eg4 {
  int i;
};
Eg4 e4;

RWBuffer<int> In : register(u1);
