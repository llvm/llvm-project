// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl -ast-dump -o - %s | FileCheck %s

// CHECK: VarDecl {{.*}} res1D '__hlsl_resource_t 
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(SRV)]] [[hlsl::resource_dimension(1D)]]
__hlsl_resource_t [[hlsl::resource_class(SRV)]] [[hlsl::dimension("1D")]] res1D;

// CHECK: VarDecl 0x{{[0-9a-f]+}} {{.*}} res2D '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(SRV)]] [[hlsl::resource_dimension(2D)]]
__hlsl_resource_t [[hlsl::resource_class(SRV)]] [[hlsl::dimension("2D")]] res2D;

// CHECK: VarDecl 0x{{[0-9a-f]+}} {{.*}} res3D '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(SRV)]] [[hlsl::resource_dimension(3D)]]
__hlsl_resource_t [[hlsl::resource_class(SRV)]] [[hlsl::dimension("3D")]] res3D;

// CHECK: VarDecl 0x{{[0-9a-f]+}} {{.*}} resCube '__hlsl_resource_t
// CHECK-SAME{LITERAL}: [[hlsl::resource_class(SRV)]] [[hlsl::resource_dimension(Cube)]]
__hlsl_resource_t [[hlsl::resource_class(SRV)]] [[hlsl::dimension("Cube")]] resCube;
