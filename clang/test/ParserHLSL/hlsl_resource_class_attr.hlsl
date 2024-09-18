// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} {{.*}} struct MyBuffer definition
// CHECK: FieldDecl 0x{{[0-9a-f]+}} <line:6:3, col:51> col:51 h '__hlsl_resource_t {{\[\[}}hlsl::resource_class(UAV)]]':'__hlsl_resource_t'
struct MyBuffer {
  __hlsl_resource_t [[hlsl::resource_class(UAV)]] h;
};

// CHECK: VarDecl 0x{{[0-9a-f]+}} <line:10:1, col:49> col:49 res '__hlsl_resource_t {{\[\[}}hlsl::resource_class(SRV)]]':'__hlsl_resource_t'
__hlsl_resource_t [[hlsl::resource_class(SRV)]] res;

// CHECK: FunctionDecl 0x{{[0-9a-f]+}} <line:14:1, line:16:1> line:14:6 f 'void ()
// CHECK: VarDecl 0x{{[0-9a-f]+}} <col:3, col:55> col:55 r '__hlsl_resource_t {{\[\[}}hlsl::resource_class(Sampler)]]':'__hlsl_resource_t'
void f() {
  __hlsl_resource_t [[hlsl::resource_class(Sampler)]] r;
}

// CHECK: ClassTemplateDecl 0x{{[0-9a-f]+}} <line:23:1, line:25:1> line:23:29 MyBuffer2
// CHECK: TemplateTypeParmDecl 0x{{[0-9a-f]+}} <col:10, col:19> col:19 typename depth 0 index 0 T
// CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} <col:22, line:25:1> line:23:29 struct MyBuffer2 definition
// CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} <col:22, col:29> col:29 implicit struct MyBuffer2
// CHECK: FieldDecl 0x{{[0-9a-f]+}} <line:24:3, col:51> col:51 h '__hlsl_resource_t {{\[\[}}hlsl::resource_class(UAV)]]':'__hlsl_resource_t'
template<typename T> struct MyBuffer2 {
  __hlsl_resource_t [[hlsl::resource_class(UAV)]] h;
};

// CHECK: ClassTemplateSpecializationDecl 0x{{[0-9a-f]+}} <line:23:1, line:25:1> line:23:29 struct MyBuffer2 definition implicit_instantiation
// CHECK: TemplateArgument type 'float'
// CHECK: BuiltinType 0x{{[0-9a-f]+}} 'float'
// CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} <col:22, col:29> col:29 implicit struct MyBuffer2
// CHECK: FieldDecl 0x{{[0-9a-f]+}} <line:24:3, col:51> col:51 h '__hlsl_resource_t {{\[\[}}hlsl::resource_class(UAV)]]':'__hlsl_resource_t'
MyBuffer2<float> myBuffer2;
