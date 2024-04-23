// RUN: %clang_cc1 -triple dxil-unknown-shadermodel6.3-library -S -finclude-default-header  -ast-dump  -x hlsl %s | FileCheck %s


// CHECK: HLSLBufferDecl {{.*}} cbuffer A
cbuffer A
{
    // CHECK-NEXT: VarDecl {{.*}} C1 'float4'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 0
    float4 C1 : packoffset(c);
    // CHECK-NEXT: VarDecl {{.*}} col:11 C2 'float'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 4
    float C2 : packoffset(c1);
    // CHECK-NEXT: VarDecl {{.*}} col:11 C3 'float'
    // CHECK-NEXT: HLSLPackOffsetAttr {{.*}} 5
    float C3 : packoffset(c1.y);
}
