// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -ast-dump %s | FileCheck %s

// Single resource field in struct

// CHECK: CXXRecordDecl {{.*}} struct A
// CHECK: FieldDecl {{.*}} Buf 'RWBuffer<float>':'hlsl::RWBuffer<float>'
struct A {
  RWBuffer<float> Buf;
};

// CHECK: VarDecl {{.*}} implicit a1.Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} a1 'hlsl_constant A'
// CHECK: HLSLResourceBindingAttr {{.*}} "u0" "space0"
// CHECK-NEXT: HLSLAssociatedResourceDeclAttr {{.*}} 'a1.Buf' 'hlsl::RWBuffer<float>'
A a1 : register(u0);

// Resource array in struct

// CHECK: CXXRecordDecl {{.*}} struct B
// CHECK: FieldDecl {{.*}} Bufs 'RWBuffer<float>[10]'
struct B {
  RWBuffer<float> Bufs[10];
};

// CHECK: VarDecl {{.*}} implicit b1.Bufs 'hlsl::RWBuffer<float>[10]'
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} b1 'hlsl_constant B'
// CHECK: HLSLResourceBindingAttr {{.*}} "u2" "space0"
// CHECK-NEXT: HLSLAssociatedResourceDeclAttr {{.*}} 'b1.Bufs' 'hlsl::RWBuffer<float>[10]'
B b1 : register(u2);

// Inheritance

// CHECK: CXXRecordDecl {{.*}} struct C
// CHECK: FieldDecl {{.*}} Buf2 'RWBuffer<float>':'hlsl::RWBuffer<float>'
struct C : A {
  RWBuffer<float> Buf2;
};

// CHECK: VarDecl {{.*}} implicit c1.A::Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} implicit c1.Buf2 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} c1 'hlsl_constant C'
// CHECK: HLSLResourceBindingAttr {{.*}} "u3" "space0"
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'c1.A::Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'c1.Buf2' 'hlsl::RWBuffer<float>'
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

// CHECK: VarDecl {{.*}} d1 'hlsl_constant D'
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
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} implicit f.a.Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} implicit f.SrvBuf 'hlsl::StructuredBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} implicit f.Samp 'hlsl::SamplerState' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} f 'hlsl_constant F'
// CHECK: HLSLResourceBindingAttr {{.*}} "t0" "space0"
// CHECK: HLSLResourceBindingAttr {{.*}} "u20" "space0"
// CHECK: HLSLResourceBindingAttr {{.*}} "s3" "space0"
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'f.E::SrvBuf' 'hlsl::StructuredBuffer<int>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'f.a.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'f.SrvBuf' 'hlsl::StructuredBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'f.Samp' 'hlsl::SamplerState'
F f : register(t0) : register(u20) : register(s3);

// Array of structs with resources

// CHECK: VarDecl {{.*}} implicit arrayOfA.0.Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} implicit arrayOfA.1.Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"
A arrayOfA[2] : register(u0, space1);

// CHECK: CXXRecordDecl {{.*}} struct G
// CHECK: FieldDecl {{.*}} multiArray 'A[2][2]'
struct G {
  A multiArray[2][2];
};

// CHECK: VarDecl {{.*}} implicit gArray.0.multiArray.0.0.Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} implicit gArray.0.multiArray.0.1.Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} implicit gArray.0.multiArray.1.0.Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} implicit gArray.0.multiArray.1.1.Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} implicit gArray.1.multiArray.0.0.Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} implicit gArray.1.multiArray.0.1.Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} implicit gArray.1.multiArray.1.0.Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} implicit gArray.1.multiArray.1.1.Buf 'hlsl::RWBuffer<float>' callinit
// CHECK: HLSLResourceBindingAttr {{.*}} Implicit "" "0"

// CHECK: VarDecl {{.*}} gArray 'hlsl_constant G[2]'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.0.multiArray.0.0.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.0.multiArray.0.1.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.0.multiArray.1.0.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.0.multiArray.1.1.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.1.multiArray.0.0.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.1.multiArray.0.1.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.1.multiArray.1.0.Buf' 'hlsl::RWBuffer<float>'
// CHECK: HLSLAssociatedResourceDeclAttr {{.*}} 'gArray.1.multiArray.1.1.Buf' 'hlsl::RWBuffer<float>'
G gArray[2] : register(u10, space2);

// Static struct with resources

// CHECK-NOT: VarDecl {{.*}} a2.Buf
// CHECK: VarDecl {{.*}} a2 'hlsl_private A' static cinit
static A a2 = { a1 };
