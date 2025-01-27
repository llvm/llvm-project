// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -ast-dump -o - %s | FileCheck %s

// CHECK: CXXRecordDecl {{.*}} struct EmptyStruct definition
struct EmptyStruct {
};

// CHECK: NamespaceDecl {{.*}} NS1
namespace NS1 {
  // CHECK: CXXRecordDecl {{.*}} struct Foo definition
  struct Foo { 
    float a;
    EmptyStruct es;
  };
  
  // CHECK: CXXRecordDecl {{.*}} struct Bar definition
  struct Bar {
    // CHECK: CXXRecordDecl {{.*}} struct Foo definition
    struct Foo {
      int b;
      EmptyStruct es;
    };
    // CHECK: CXXRecordDecl {{.*}} implicit class __layout_Foo definition
    // CHECK: FieldDecl {{.*}} b 'int'
  };
  // CHECK: CXXRecordDecl {{.*}} implicit class __layout_Foo definition
  // CHECK: FieldDecl {{.*}} a 'float'
}

struct Foo {
  double c;
  EmptyStruct es;
};

// CHECK: HLSLBufferDecl {{.*}}  line:37:9 cbuffer CB1
// CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
// CHECK: HLSLResourceAttr {{.*}} Implicit CBuffer
cbuffer CB1 {
  // CHECK: VarDecl {{.*}} foo1 'hlsl_constant Foo'
  Foo foo1;
  // CHECK: VarDecl {{.*}} foo2 'hlsl_constant NS1::Foo'
  NS1::Foo foo2;
  // CHECK: VarDecl {{.*}} foo3 'hlsl_constant NS1::Bar::Foo'
  NS1::Bar::Foo foo3;
  // CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_CB1 definition
  // CHECK: FieldDecl {{.*}} foo1 '__layout_Foo'
  // CHECK: FieldDecl {{.*}} foo2 'NS1::__layout_Foo'
  // CHECK: FieldDecl {{.*}} foo3 'NS1::Bar::__layout_Foo'
}
// CHECK: CXXRecordDecl {{.*}} implicit class __layout_Foo definition
// CHECK: FieldDecl {{.*}} c 'double'

struct CB1ExpectedShape {
    double a1;
    float a2;
    int a;
};
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(CB1ExpectedShape, __layout_CB1), "");

namespace NS2 {
  struct Foo { 
    float d[4];
    EmptyStruct es;
  };
  // CHECK: HLSLBufferDecl {{.*}} line:67:11 cbuffer CB2
  // CHECK: HLSLResourceClassAttr {{.*}} Implicit CBuffer
  // CHECK: HLSLResourceAttr {{.*}} Implicit CBuffer
  cbuffer CB2 {
    // CHECK: VarDecl {{.*}} foo0 'hlsl_constant ::Foo':'hlsl_constant Foo'
    ::Foo foo0;
    // CHECK: VarDecl {{.*}} foo1 'hlsl_constant Foo':'hlsl_constant NS2::Foo'
    Foo foo1;
    // CHECK: VarDecl {{.*}} foo2 'hlsl_constant NS1::Foo'
    NS1::Foo foo2;
    // CHECK: VarDecl {{.*}} foo3 'hlsl_constant NS1::Bar::Foo'
    NS1::Bar::Foo foo3;
    // CHECK: CXXRecordDecl {{.*}} implicit referenced class __layout_CB2 definition
    // CHECK: FieldDecl {{.*}} foo0 '__layout_Foo'
    // CHECK: FieldDecl {{.*}} foo1 'NS2::__layout_Foo'
    // CHECK: FieldDecl {{.*}} foo2 'NS1::__layout_Foo'
    // CHECK: FieldDecl {{.*}} foo3 'NS1::Bar::__layout_Foo'
  }
  // CHECK: CXXRecordDecl {{.*}} implicit class __layout_Foo definition
  // CHECK: FieldDecl {{.*}} d 'float[4]'
}

struct CB2ExpectedShape {
    double a1;
    float d[4];
    float a2;
    int a;
};
_Static_assert(__builtin_hlsl_is_scalarized_layout_compatible(CB2ExpectedShape, NS2::__layout_CB2), "");

// Add uses for the constant buffer declarations so they are not optimized away
// CHECK: ExportDecl
export float f() {
  return foo2.a + NS2::foo2.a;
}
