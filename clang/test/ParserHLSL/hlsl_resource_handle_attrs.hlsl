// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: ClassTemplateSpecializationDecl {{.*}} class RWBuffer definition implicit_instantiation
// CHECK: TemplateArgument type 'float'
// CHECK: BuiltinType {{.*}} 'float'
// CHECK: FieldDecl {{.*}} implicit referenced __handle '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(UAV)]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(float)]]
RWBuffer<float> Buffer1;

// CHECK: ClassTemplateSpecializationDecl {{.*}} class RasterizerOrderedBuffer definition implicit_instantiation
// CHECK: TemplateArgument type 'vector<float, 4>'
// CHECK: ExtVectorType {{.*}} 'vector<float, 4>' 4
// CHECK: BuiltinType {{.*}} 'float'
// CHECK: FieldDecl {{.*}} implicit referenced __handle '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(UAV)]
// CHECK-SAME{LITERAL}: [[hlsl::is_rov]]
// CHECK-SAME{LITERAL}: [[hlsl::contained_type(vector<float, 4>)]]
RasterizerOrderedBuffer<vector<float, 4> > BufferArray3[4];
