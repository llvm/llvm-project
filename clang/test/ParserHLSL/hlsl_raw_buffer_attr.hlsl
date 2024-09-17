// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} {{.*}} struct MyBuffer definition
// CHECK: FieldDecl 0x{{[0-9a-f]+}} <line:6:3, col:72> col:72 h1 '__hlsl_resource_t {{\[\[}}hlsl::resource_class(UAV)]] {{\[\[}}hlsl::raw_buffer]]':'__hlsl_resource_t'
struct MyBuffer {
  __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::raw_buffer]] h1;
};

// CHECK: VarDecl 0x{{[0-9a-f]+}} <line:10:1, col:70> col:70 h2 '__hlsl_resource_t {{\[\[}}hlsl::resource_class(SRV)]] {{\[\[}}hlsl::raw_buffer]]':'__hlsl_resource_t'
__hlsl_resource_t [[hlsl::raw_buffer]] [[hlsl::resource_class(SRV)]] h2;

// CHECK: FunctionDecl 0x{{[0-9a-f]+}} <line:14:1, line:16:1> line:14:6 f 'void ()
// CHECK: VarDecl 0x{{[0-9a-f]+}} <col:3, col:72> col:72 h3 '__hlsl_resource_t {{\[\[}}hlsl::resource_class(UAV)]] {{\[\[}}hlsl::raw_buffer]]':'__hlsl_resource_t'
void f() {
  __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::raw_buffer]] h3;
}
