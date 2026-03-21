// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -ast-dump %s | FileCheck %s -check-prefixes=CHECK,DXIL
// RUN: %clang_cc1 -triple spirv-unknown-vulkan-compute -ast-dump %s | FileCheck %s -check-prefixes=CHECK,SPIRV

// Single resource field in struct

// CHECK: CXXRecordDecl {{.*}} struct A
// CHECK: FieldDecl {{.*}} Buf 'RWBuffer<float>':'hlsl::RWBuffer<float>'
struct A {
  RWBuffer<float> Buf;
};

// CHECK: VarDecl {{.*}} implicit a1.Buf 'hlsl::RWBuffer<float>' callinit
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 0 0
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u0" "space0"

// CHECK: VarDecl {{.*}} a1 'A'
// CHECK: HLSLVkBindingAttr {{.*}} 0 0
// CHECK: HLSLResourceBindingAttr {{.*}} "u0" "space0"
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'a1.Buf' 'hlsl::RWBuffer<float>'
[[vk::binding(0)]]
A a1 : register(u0);

// Resource array in struct

// CHECK: CXXRecordDecl {{.*}} struct B
// CHECK: FieldDecl {{.*}} Bufs 'RWBuffer<float>[10]'
struct B {
  RWBuffer<float> Bufs[10];
};

// Check when the struct has only [[vk::binding]] binding attribute.

// CHECK: VarDecl {{.*}} b1.Bufs 'hlsl::RWBuffer<float>[10]'
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "" "0"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 2 0

// CHECK: VarDecl {{.*}} b1 'B'
// CHECK: HLSLVkBindingAttr {{.*}} 2 0
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'b1.Bufs' 'hlsl::RWBuffer<float>[10]'
[[vk::binding(2)]]
B b1;

// Inheritance

// CHECK: CXXRecordDecl {{.*}} struct C
// CHECK: FieldDecl {{.*}} Buf2 'RWBuffer<float>':'hlsl::RWBuffer<float>'
struct C : A {
  RWBuffer<float> Buf2;
};

// CHECK: VarDecl {{.*}} implicit c1.A::Buf 'hlsl::RWBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u3" "space0"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 3 0

// CHECK: VarDecl {{.*}} implicit c1.Buf2 'hlsl::RWBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u4" "space0"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 4 0

// CHECK: VarDecl {{.*}} c1 'C'
// CHECK: HLSLVkBindingAttr {{.*}} 3 0
// CHECK: HLSLResourceBindingAttr {{.*}} "u3" "space0"
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'c1.A::Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'c1.Buf2' 'hlsl::RWBuffer<float>'
[[vk::binding(3)]]
C c1 : register(u3);

// Inheritance with same named field
// CHECK: CXXRecordDecl {{.*}} struct D
// CHECK: FieldDecl {{.*}} A 'A'
struct D : A {
    A A;
};

// CHECK: VarDecl {{.*}} implicit d1.A::Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} implicit d1.A.Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} d1 'D'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'd1.A::Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'd1.A.Buf' 'hlsl::RWBuffer<float>'
D d1;

// Inheritance and Multiple Resources Kinds

// CHECK: CXXRecordDecl {{.*}} class E
// CHECK: FieldDecl {{.*}} SrvBuf 'StructuredBuffer<int>':'hlsl::StructuredBuffer<int>'
class E {
  StructuredBuffer<int> SrvBuf;
};

// CHECK: CXXRecordDecl {{.*}} class F
// CHECK: FieldDecl {{.*}} a 'A'
// CHECK: FieldDecl {{.*}} SrvBuf 'StructuredBuffer<float>':'hlsl::StructuredBuffer<float>'
// CHECK: FieldDecl {{.*}} Samp 'SamplerState'
class F : E {
  A a;
  StructuredBuffer<float> SrvBuf;
  SamplerState Samp;
};

// CHECK: VarDecl {{.*}} implicit f.E::SrvBuf 'hlsl::StructuredBuffer<int>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "t0" "space0"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 10 0

// CHECK: VarDecl {{.*}} implicit f.a.Buf 'hlsl::RWBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u20" "space0"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 11 0

// CHECK: VarDecl {{.*}} implicit f.SrvBuf 'hlsl::StructuredBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "t1" "space0"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 12 0

// CHECK: VarDecl {{.*}} implicit f.Samp 'hlsl::SamplerState' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "s3" "space0"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 13 0

// CHECK: VarDecl {{.*}} f 'F'
// CHECK: HLSLVkBindingAttr {{.*}} 10 0
// CHECK: HLSLResourceBindingAttr {{.*}} "t0" "space0"
// CHECK: HLSLResourceBindingAttr {{.*}} "u20" "space0"
// CHECK: HLSLResourceBindingAttr {{.*}} "s3" "space0"
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'f.E::SrvBuf' 'hlsl::StructuredBuffer<int>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'f.a.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'f.SrvBuf' 'hlsl::StructuredBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'f.Samp' 'hlsl::SamplerState'
[[vk::binding(10)]]
F f : register(t0) : register(u20) : register(s3);

// Array of structs with resources

// CHECK: VarDecl {{.*}} implicit arrayOfA.0.Buf 'hlsl::RWBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u0" "space1"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 0 1

// CHECK: VarDecl {{.*}} implicit arrayOfA.1.Buf 'hlsl::RWBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u1" "space1"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 1 1

// CHECK: VarDecl {{.*}} arrayOfA 'A[2]'
// CHECK: HLSLVkBindingAttr {{.*}} 0 1
// CHECK: HLSLResourceBindingAttr {{.*}} "u0" "space1"
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'arrayOfA.0.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'arrayOfA.1.Buf' 'hlsl::RWBuffer<float>'
[[vk::binding(0, 1)]]
A arrayOfA[2] : register(u0, space1);

// CHECK: CXXRecordDecl {{.*}} struct G
// CHECK: FieldDecl {{.*}} multiArray 'A[2][2]'
struct G {
  A multiArray[2][2];
};

// CHECK: VarDecl {{.*}} implicit gArray.0.multiArray.0.0.Buf 'hlsl::RWBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u10" "space2"
// SPRIV: HLSLVkBindingAttr {{.*}} Implicit 10 2

// CHECK: VarDecl {{.*}} implicit gArray.0.multiArray.0.1.Buf 'hlsl::RWBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u11" "space2"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 11 2

// CHECK: VarDecl {{.*}} implicit gArray.0.multiArray.1.0.Buf 'hlsl::RWBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u12" "space2"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 12 2
// CHECK: VarDecl {{.*}} implicit gArray.0.multiArray.1.1.Buf 'hlsl::RWBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u13" "space2"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 13 2

// CHECK: VarDecl {{.*}} implicit gArray.1.multiArray.0.0.Buf 'hlsl::RWBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u14" "space2"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 14 2

// CHECK: VarDecl {{.*}} implicit gArray.1.multiArray.0.1.Buf 'hlsl::RWBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u15" "space2"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 15 2

// CHECK: VarDecl {{.*}} implicit gArray.1.multiArray.1.0.Buf 'hlsl::RWBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u16" "space2"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 16 2

// CHECK: VarDecl {{.*}} implicit gArray.1.multiArray.1.1.Buf 'hlsl::RWBuffer<float>' callinit
// DXIL: HLSLResourceBindingAttr {{.*}} Implicit "u17" "space2"
// SPIRV: HLSLVkBindingAttr {{.*}} Implicit 17 2

// CHECK: VarDecl {{.*}} gArray 'G[2]'
// CHECK: HLSLVkBindingAttr {{.*}} 10 2
// CHECK: HLSLResourceBindingAttr {{.*}} "u10" "space2"
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.0.multiArray.0.0.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.0.multiArray.0.1.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.0.multiArray.1.0.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.0.multiArray.1.1.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.1.multiArray.0.0.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.1.multiArray.0.1.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.1.multiArray.1.0.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.1.multiArray.1.1.Buf' 'hlsl::RWBuffer<float>'
[[vk::binding(10, 2)]]
G gArray[2] : register(u10, space2);

// Static struct with resources

// CHECK-NOT: VarDecl {{.*}} a2.Buf
// CHECK: VarDecl {{.*}} a2 'hlsl_private A' static cinit
static A a2 = { a1 };
