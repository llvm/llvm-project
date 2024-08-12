// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: -ClassTemplateDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> implicit RWBuffer
// CHECK: -CXXRecordDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> implicit class RWBuffer definition
// CHECK: -FieldDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> implicit h 'element_type *'
// CHECK: -HLSLResourceClassAttr 0x{{[0-9a-f]+}} <<invalid sloc>> Implicit UAV
// CHECK: -HLSLResourceAttr 0x{{[0-9a-f]+}} <<invalid sloc>> Implicit TypedBuffer
RasterizerOrderedBuffer<vector<float, 4> > BufferArray3[4] : register(u4, space1);

// CHECK: -ClassTemplateSpecializationDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> class RWBuffer definition implicit_instantiation
// CHECK: -FieldDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> implicit referenced h 'float *'
// CHECK: -HLSLResourceClassAttr 0x{{[0-9a-f]+}} <<invalid sloc>> Implicit UAV
// CHECK: -HLSLResourceAttr 0x{{[0-9a-f]+}} <<invalid sloc>> Implicit TypedBuffer
RWBuffer<float> Buffer1;
