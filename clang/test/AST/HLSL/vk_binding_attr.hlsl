// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-library -finclude-default-header -ast-dump -o - %s | FileCheck %s -check-prefixes=SPV,CHECK
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-library -finclude-default-header -ast-dump -o - %s | FileCheck %s -check-prefixes=DXIL,CHECK

// CHECK: VarDecl {{.*}} Buf 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>'
// SPV-NEXT: CXXConstructExpr {{.*}} 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>' 'void (unsigned int, unsigned int, int, unsigned int, const char *)'
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 23
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 102
// DXIL-NEXT: CXXConstructExpr {{.*}} 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>' 'void (unsigned int, int, unsigned int, unsigned int, const char *)'
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// DXIL-NEXT: IntegerLiteral {{.*}} 'int' 1
// SPV: HLSLVkBindingAttr {{.*}} 23 102
// DXIL-NOT: HLSLVkBindingAttr
[[vk::binding(23, 102)]] StructuredBuffer<float> Buf;

// CHECK: VarDecl {{.*}} Buf2 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>' 'void (unsigned int, unsigned int, int, unsigned int, const char *)'
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 14
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 23
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 102
// SPV: HLSLVkBindingAttr {{.*}} 14 1
// DXIL-NOT: HLSLVkBindingAttr
// CHECK: HLSLResourceBindingAttr {{.*}} "t23" "space102"
[[vk::binding(14, 1)]] StructuredBuffer<float> Buf2 : register(t23, space102);

// CHECK: VarDecl {{.*}} Buf3 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>' 'void (unsigned int, unsigned int, int, unsigned int, const char *)'
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 14
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 23
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 102
// SPV: HLSLVkBindingAttr {{.*}} 14 0
// DXIL-NOT: HLSLVkBindingAttr
// CHECK: HLSLResourceBindingAttr {{.*}} "t23" "space102"
[[vk::binding(14)]] StructuredBuffer<float> Buf3 : register(t23, space102);
 
// CHECK: HLSLBufferDecl {{.*}} cbuffer CB
// CHECK-NEXT: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// SPV-NEXT: HLSLVkBindingAttr {{.*}} 1 2
// DXIL-NOT: HLSLVkBindingAttr
[[vk::binding(1, 2)]] cbuffer CB {
  float a;
}

// CHECK: VarDecl {{.*}} Buf4 'Buffer<int>':'hlsl::Buffer<int>'
// SPV-NEXT: CXXConstructExpr {{.*}} 'Buffer<int>':'hlsl::Buffer<int>' 'void (unsigned int, unsigned int, int, unsigned int, const char *)'
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 24
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 103
// DXL-NEXT: CXXConstructExpr {{.*}} 'Buffer<int>':'hlsl::Buffer<int>' 'void (unsigned int, int, unsigned int, unsigned int, const char *)'
// SPV: HLSLVkBindingAttr {{.*}} 24 103
// DXIL-NOT: HLSLVkBindingAttr
[[vk::binding(24, 103)]] Buffer<int> Buf4;

// CHECK: VarDecl {{.*}} Buf5 'RWBuffer<int2>':'hlsl::RWBuffer<vector<int, 2>>'
// SPV-NEXT: CXXConstructExpr {{.*}} 'RWBuffer<int2>':'hlsl::RWBuffer<vector<int, 2>>' 'void (unsigned int, unsigned int, int, unsigned int, const char *)'
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 25
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 104
// DXL-NEXT: CXXConstructExpr {{.*}} 'Buffer<int>':'hlsl::Buffer<int>' 'void (unsigned int, int, unsigned int, unsigned int, const char *)'
// SPV: HLSLVkBindingAttr {{.*}} 25 104
// DXIL-NOT: HLSLVkBindingAttr
[[vk::binding(25, 104)]] RWBuffer<int2> Buf5;

// CHECK: VarDecl {{.*}} Buf6 'RWStructuredBuffer<int>':'hlsl::RWStructuredBuffer<int>'
// SPV-NEXT: CXXConstructExpr {{.*}} 'RWStructuredBuffer<int>':'hlsl::RWStructuredBuffer<int>' 'void (unsigned int, unsigned int, int, unsigned int, const char *)'
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 26
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 105
// DXL-NEXT: CXXConstructExpr {{.*}} 'Buffer<int>':'hlsl::Buffer<int>' 'void (unsigned int, int, unsigned int, unsigned int, const char *)'
// SPV: HLSLVkBindingAttr {{.*}} 26 105
// DXIL-NOT: HLSLVkBindingAttr
[[vk::binding(26, 105)]] RWStructuredBuffer<int> Buf6;
