// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} {{.*}} struct MyBuffer definition
// CHECK: FieldDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 3]]:3, col:51> col:51 h '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(UAV)]]
struct MyBuffer {
  __hlsl_resource_t [[hlsl::resource_class(UAV)]] h;
};

// CHECK: VarDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 2]]:1, col:49> col:49 res '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(SRV)]]
__hlsl_resource_t [[hlsl::resource_class(SRV)]] res;

// CHECK: FunctionDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 3]]:1, line:[[# @LINE + 5]]:1> line:[[# @LINE + 3]]:6 f 'void ()
// CHECK: VarDecl 0x{{[0-9a-f]+}} <col:3, col:55> col:55 r '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(Sampler)]]
void f() {
  __hlsl_resource_t [[hlsl::resource_class(Sampler)]] r;
}

// CHECK: ClassTemplateDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 6]]:1, line:[[# @LINE + 8]]:1> line:[[# @LINE + 6]]:29 MyBuffer2
// CHECK: TemplateTypeParmDecl 0x{{[0-9a-f]+}} <col:10, col:19> col:19 typename depth 0 index 0 T
// CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} <col:22, line:[[# @LINE + 6]]:1> line:[[# @LINE + 4]]:29 struct MyBuffer2 definition
// CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} <col:22, col:29> col:29 implicit struct MyBuffer2
// CHECK: FieldDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 3]]:3, col:51> col:51 h '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(UAV)]]
template<typename T> struct MyBuffer2 {
  __hlsl_resource_t [[hlsl::resource_class(UAV)]] h;
};

// CHECK: ClassTemplateSpecializationDecl 0x{{[0-9a-f]+}} <line:[[# @LINE - 4]]:1, line:[[# @LINE - 2]]:1> line:[[# @LINE - 4]]:29 struct MyBuffer2 definition implicit_instantiation
// CHECK: TemplateArgument type 'float'
// CHECK: BuiltinType 0x{{[0-9a-f]+}} 'float'
// CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} <col:22, col:29> col:29 implicit struct MyBuffer2
// CHECK: FieldDecl 0x{{[0-9a-f]+}} <line:[[# @LINE - 7]]:3, col:51> col:51 h '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(UAV)]]
MyBuffer2<float> myBuffer2;
