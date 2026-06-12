// Establish a baseline without define specified
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -ast-dump \
// RUN:  -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,NO-OVERRIDE

// Check that we can set the entry function even if it doesn't have an attr
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -ast-dump \
// RUN:  -hlsl-entry none_main -fdx-rootsignature-define=SampleCBV \
// RUN:  -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,SET

// Check that we can set the entry function overriding an attr
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -ast-dump \
// RUN:  -hlsl-entry uav_main -fdx-rootsignature-define=SampleCBV \
// RUN:  -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,OVERRIDE

// Check that we can override with a command line root signature
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-library -ast-dump \
// RUN:  -hlsl-entry cbv_main -fdx-rootsignature-define=CmdRS -DCmdRS='"SRV(t0)"' \
// RUN:  -disable-llvm-passes -o - %s | FileCheck %s --check-prefixes=CHECK,CMD

#define SampleCBV "CBV(b0)"
#define SampleUAV "UAV(u0)"

// CMD: -HLSLRootSignatureDecl 0x{{.*}} {{.*}} implicit [[CMD_DECL:__hlsl_rootsig_decl_\d*]]
// CMD-SAME: version: 1.1, RootElements{
// CMD-SAME: RootSRV(t0,
// CMD-SAME:   space = 0, visibility = All, flags = DataStaticWhileSetAtExecute
// CMD-SAME: )}

// CHECK: -HLSLRootSignatureDecl 0x{{.*}} {{.*}} implicit [[CBV_DECL:__hlsl_rootsig_decl_\d*]]
// CHECK-SAME: version: 1.1, RootElements{
// CHECK-SAME: RootCBV(b0,
// CHECK-SAME:   space = 0, visibility = All, flags = DataStaticWhileSetAtExecute
// CHECK-SAME: )}

// CHECK-LABEL: -FunctionDecl 0x{{.*}} {{.*}} cbv_main
// NO-OVERRIDE: -RootSignatureAttr 0x{{.*}} {{.*}} [[CBV_DECL]]
// SET: -RootSignatureAttr 0x{{.*}} {{.*}} [[CBV_DECL]]
// CMD: -RootSignatureAttr 0x{{.*}} {{.*}} [[CMD_DECL]]

[RootSignature(SampleCBV)]
void cbv_main() {}

// CHECK: -HLSLRootSignatureDecl 0x{{.*}} {{.*}} implicit [[UAV_DECL:__hlsl_rootsig_decl_\d*]]
// CHECK-SAME: version: 1.1, RootElements{
// CHECK-SAME: RootUAV(u0,
// CHECK-SAME:   space = 0, visibility = All, flags = DataVolatile
// CHECK-SAME: )}

// CHECK-LABEL: -FunctionDecl 0x{{.*}} {{.*}} uav_main
// NO-OVERRIDE: -RootSignatureAttr 0x{{.*}} {{.*}} [[UAV_DECL]]
// SET: -RootSignatureAttr 0x{{.*}} {{.*}} [[UAV_DECL]]
// OVERRIDE: -RootSignatureAttr 0x{{.*}} {{.*}} [[CBV_DECL]]

[RootSignature(SampleUAV)]
void uav_main() {}

// CHECK-LABEL: -FunctionDecl 0x{{.*}} {{.*}} none_main
// NO-OVERRIDE-NONE: -RootSignatureAttr
// SET: -RootSignatureAttr 0x{{.*}} {{.*}} [[CBV_DECL]]
// OVERRIDE-NONE: -RootSignatureAttr

void none_main() {}
