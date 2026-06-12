// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} {{.*}} struct MyBuffer definition
// CHECK: FieldDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 4]]:3, col:70> col:70 h '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(UAV)]]
// CHECK-SAME{LITERAL}: [[hlsl::is_array]]
struct MyBuffer {
  __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::is_array]] h;
};

// CHECK: VarDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 3]]:1, col:68> col:68 res '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(SRV)]]
// CHECK-SAME{LITERAL}: [[hlsl::is_array]]
__hlsl_resource_t [[hlsl::is_array]] [[hlsl::resource_class(SRV)]] res;

// CHECK: FunctionDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 4]]:1, line:[[# @LINE + 6]]:1> line:[[# @LINE + 4]]:6 f 'void ()
// CHECK: VarDecl 0x{{[0-9a-f]+}} <col:3, col:74> col:74 r '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(Sampler)]]
// CHECK-SAME{LITERAL}: [[hlsl::is_array]]
void f() {
  __hlsl_resource_t [[hlsl::resource_class(Sampler)]] [[hlsl::is_array]] r;
}
