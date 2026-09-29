// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-library -finclude-default-header -ast-dump -o - %s | FileCheck %s -check-prefixes=SPV,CHECK
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.8-library -finclude-default-header -ast-dump -o - %s | FileCheck %s -check-prefixes=DXIL,CHECK

// CHECK: VarDecl {{.*}} Buf 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>'
// CHECK-NEXT: CallExpr {{.*}} 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'hlsl::StructuredBuffer<float> (*)(unsigned int, unsigned int, int, unsigned int, const char *)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::StructuredBuffer<float> (unsigned int, unsigned int, int, unsigned int, const char *)' 
// CHECK-NEXT-SAME: CXXMethod {{.*}} '__createFromBinding' 'hlsl::StructuredBuffer<float> (unsigned int, unsigned int, int, unsigned int, const char *)'
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 23
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 102
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// CHECK: HLSLVkBindingAttr {{.*}} 23 102
[[vk::binding(23, 102)]] StructuredBuffer<float> Buf;

// CHECK: VarDecl {{.*}} Buf2 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>'
// CHECK-NEXT: CallExpr {{.*}} 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'hlsl::StructuredBuffer<float> (*)(unsigned int, unsigned int, int, unsigned int, const char *)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::StructuredBuffer<float> (unsigned int, unsigned int, int, unsigned int, const char *)' 
// CHECK-NEXT-SAME: CXXMethod {{.*}} '__createFromBinding' 'hlsl::StructuredBuffer<float> (unsigned int, unsigned int, int, unsigned int, const char *)'
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 14
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 1
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 23
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 102
// CHECK: HLSLVkBindingAttr {{.*}} 14 1
// CHECK: HLSLResourceBindingAttr {{.*}} "t23" "space102"
[[vk::binding(14, 1)]] StructuredBuffer<float> Buf2 : register(t23, space102);

// CHECK: VarDecl {{.*}} Buf3 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>'
// CHECK-NEXT: CallExpr {{.*}} 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'hlsl::StructuredBuffer<float> (*)(unsigned int, unsigned int, int, unsigned int, const char *)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'hlsl::StructuredBuffer<float> (unsigned int, unsigned int, int, unsigned int, const char *)' 
// CHECK-NEXT-SAME: CXXMethod {{.*}} '__createFromBinding' 'hlsl::StructuredBuffer<float> (unsigned int, unsigned int, int, unsigned int, const char *)'
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 14
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 23
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 102
// CHECK: HLSLVkBindingAttr {{.*}} 14 0
// CHECK: HLSLResourceBindingAttr {{.*}} "t23" "space102"
[[vk::binding(14)]] StructuredBuffer<float> Buf3 : register(t23, space102);
 
// CHECK: HLSLBufferDecl {{.*}} cbuffer CB
// CHECK-NEXT: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK: HLSLVkBindingAttr {{.*}} 1 2
[[vk::binding(1, 2)]] cbuffer CB {
  float a;
}

// CHECK: VarDecl {{.*}} Buf4 'Buffer<int>':'hlsl::Buffer<int>'
// CHECK-NEXT: CallExpr {{.*}} 'Buffer<int>':'hlsl::Buffer<int>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'hlsl::Buffer<int> (*)(unsigned int, unsigned int, int, unsigned int, const char *)' <FunctionToPointerDecay>
// SPV-NEXT: DeclRefExpr {{.*}} 'hlsl::Buffer<int> (unsigned int, unsigned int, int, unsigned int, const char *)' 
// SPV-NEXT-SAME: CXXMethod {{.*}} '__createFromBinding' 'hlsl::Buffer<int> (unsigned int, unsigned int, int, unsigned int, const char *)'
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 24
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 103
// DXIL-NEXT: DeclRefExpr {{.*}} 'hlsl::Buffer<int> (unsigned int, unsigned int, int, unsigned int, const char *)'
// DXIL-NEXT-SAME: CXXMethod {{.*}} '__createFromImplicitBinding' 'hlsl::Buffer<int> (unsigned int, unsigned int, int, unsigned int, const char *)'
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 2
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// CHECK: HLSLVkBindingAttr {{.*}} 24 103
[[vk::binding(24, 103)]] Buffer<int> Buf4;

// CHECK: VarDecl {{.*}} Buf5 'RWBuffer<int2>':'hlsl::RWBuffer<vector<int, 2>>'
// CHECK-NEXT: CallExpr {{.*}} 'RWBuffer<int2>':'hlsl::RWBuffer<vector<int, 2>>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'hlsl::RWBuffer<vector<int, 2>> (*)(unsigned int, unsigned int, int, unsigned int, const char *)' <FunctionToPointerDecay>
// SPV-NEXT: DeclRefExpr {{.*}} 'hlsl::RWBuffer<vector<int, 2>> (unsigned int, unsigned int, int, unsigned int, const char *)' 
// SPV-NEXT-SAME: CXXMethod {{.*}} '__createFromBinding' 'Buffer<int2> (unsigned int, unsigned int, int, unsigned int, const char *)'
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 25
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 104
// DXIL-NEXT: DeclRefExpr {{.*}} 'hlsl::RWBuffer<vector<int, 2>> (unsigned int, unsigned int, int, unsigned int, const char *)' 
// DXIL-NEXT-SAME: CXXMethod {{.*}} '__createFromImplicitBinding' 'Buffer<int2> (unsigned int, unsigned int, int, unsigned int, const char *)'
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 3
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// CHECK: HLSLVkBindingAttr {{.*}} 25 104
[[vk::binding(25, 104)]] RWBuffer<int2> Buf5;

// CHECK: VarDecl {{.*}} Buf6 'RWStructuredBuffer<int>':'hlsl::RWStructuredBuffer<int>'
// CHECK-NEXT: CallExpr {{.*}} 'RWStructuredBuffer<int>':'hlsl::RWStructuredBuffer<int>'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'hlsl::RWStructuredBuffer<int> (*)(unsigned int, unsigned int, int, unsigned int, const char *, unsigned int)' <FunctionToPointerDecay>
// SPV-NEXT: DeclRefExpr {{.*}} 'hlsl::RWStructuredBuffer<int> (unsigned int, unsigned int, int, unsigned int, const char *, unsigned int)' 
// SPV-NEXT-SAME: CXXMethod {{.*}} '__createFromBindingwithImplicitCounter' 'hlsl::RWStructuredBuffer<int> (unsigned int, unsigned int, int, unsigned int, const char *, unsigned int)'
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 26
// SPV-NEXT: IntegerLiteral {{.*}} 'unsigned int' 105
// DXIL-NEXT: DeclRefExpr {{.*}} 'hlsl::RWStructuredBuffer<int> (unsigned int, unsigned int, int, unsigned int, const char *, unsigned int)' 
// DXIL-NEXT-SAME: CXXMethod {{.*}} '__createFromImplicitBindingwithImplicitCounter' 'hlsl::RWStructuredBuffer<int> (unsigned int, unsigned int, int, unsigned int, const char *, unsigned int)'
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 4
// DXIL-NEXT: IntegerLiteral {{.*}} 'unsigned int' 0
// CHECK: HLSLVkBindingAttr {{.*}} 26 105
[[vk::binding(26, 105)]] RWStructuredBuffer<int> Buf6;
