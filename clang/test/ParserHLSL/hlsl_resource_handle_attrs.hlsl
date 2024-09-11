// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: -ClassTemplateSpecializationDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> class RWBuffer definition implicit_instantiation
// CHECK: -TemplateArgument type 'float'
// CHECK: `-BuiltinType 0x{{[0-9a-f]+}} 'float'
// CHECK: -FieldDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> implicit referenced h 'float * {{\[\[}}hlsl::resource_class(UAV)]]':'float *'
// CHECK: -HLSLResourceAttr 0x{{[0-9a-f]+}} <<invalid sloc>> Implicit TypedBuffer
RWBuffer<float> Buffer1;

// CHECK: -ClassTemplateSpecializationDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> class RasterizerOrderedBuffer definition implicit_instantiation
// CHECK: -TemplateArgument type 'vector<float, 4>'
// CHECK: `-ExtVectorType 0x{{[0-9a-f]+}} 'vector<float, 4>' 4
// CHECK: `-BuiltinType 0x{{[0-9a-f]+}} 'float'
// CHECK: -FieldDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> implicit referenced h 'vector<float *, 4> {{\[\[}}hlsl::resource_class(UAV)]] {{\[\[}}hlsl::is_rov]]':'vector<float *, 4>'
// CHECK: -HLSLResourceAttr 0x{{[0-9a-f]+}} <<invalid sloc>> Implicit TypedBuffer
RasterizerOrderedBuffer<vector<float, 4> > BufferArray3[4];
