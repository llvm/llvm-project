// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} {{.*}} struct MyBuffer definition
// CHECK: FieldDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 4]]:3, col:67> col:67 h '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(UAV)]]
// CHECK-SAME{LITERAL}: [[hlsl::is_ms]]
struct MyBuffer {
  __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::is_ms]] h;
};

// CHECK: VarDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 3]]:1, col:65> col:65 res '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(SRV)]]
// CHECK-SAME{LITERAL}: [[hlsl::is_ms]]
__hlsl_resource_t [[hlsl::is_ms]] [[hlsl::resource_class(SRV)]] res;

// CHECK: FunctionDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 4]]:1, line:[[# @LINE + 6]]:1> line:[[# @LINE + 4]]:6 f 'void ()
// CHECK: VarDecl 0x{{[0-9a-f]+}} <col:3, col:71> col:71 r '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(Sampler)]]
// CHECK-SAME{LITERAL}: [[hlsl::is_ms]]
void f() {
  __hlsl_resource_t [[hlsl::resource_class(Sampler)]] [[hlsl::is_ms]] r;
}
