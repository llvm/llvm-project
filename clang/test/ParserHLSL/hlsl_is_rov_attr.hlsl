// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: CXXRecordDecl 0x{{[0-9a-f]+}} {{.*}} struct MyBuffer definition
// CHECK: FieldDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 4]]:3, col:68> col:68 h '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(UAV)]]
// CHECK-SAME{LITERAL}: [[hlsl::is_rov]]
struct MyBuffer {
  __hlsl_resource_t [[hlsl::resource_class(UAV)]] [[hlsl::is_rov]] h;
};

// CHECK: VarDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 3]]:1, col:66> col:66 res '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(SRV)]]
// CHECK-SAME{LITERAL}: [[hlsl::is_rov]]
__hlsl_resource_t [[hlsl::is_rov]] [[hlsl::resource_class(SRV)]] res;

// CHECK: FunctionDecl 0x{{[0-9a-f]+}} <line:[[# @LINE + 4]]:1, line:[[# @LINE + 6]]:1> line:[[# @LINE + 4]]:6 f 'void ()
// CHECK: VarDecl 0x{{[0-9a-f]+}} <col:3, col:72> col:72 r '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(Sampler)]]
// CHECK-SAME{LITERAL}: [[hlsl::is_rov]]
void f() {
  __hlsl_resource_t [[hlsl::resource_class(Sampler)]] [[hlsl::is_rov]] r;
}
