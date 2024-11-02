// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: -ClassTemplateSpecializationDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> class RWBuffer definition implicit_instantiation
// CHECK: -TemplateArgument type 'float'
// CHECK: `-BuiltinType 0x{{[0-9a-f]+}} 'float'
// CHECK: -FieldDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> implicit h '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(UAV)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(float)]]
// CHECK-SAME: ':'__hlsl_resource_t'
// CHECK: -HLSLResourceAttr 0x{{[0-9a-f]+}} <<invalid sloc>> Implicit TypedBuffer
RWBuffer<float> Buffer1;

// CHECK: -ClassTemplateSpecializationDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> class RasterizerOrderedBuffer definition implicit_instantiation
// CHECK: -TemplateArgument type 'vector<float, 4>'
// CHECK: `-ExtVectorType 0x{{[0-9a-f]+}} 'vector<float, 4>' 4
// CHECK: `-BuiltinType 0x{{[0-9a-f]+}} 'float'
// CHECK: -FieldDecl 0x{{[0-9a-f]+}} <<invalid sloc>> <invalid sloc> implicit h '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(UAV)]
// CHECK-SAME{LITERAL}: [[hlsl::is_rov]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<float, 4>)]]
// CHECK-SAME: ':'__hlsl_resource_t'
// CHECK: -HLSLResourceAttr 0x{{[0-9a-f]+}} <<invalid sloc>> Implicit TypedBuffer
RasterizerOrderedBuffer<vector<float, 4> > BufferArray3[4];
