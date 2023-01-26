// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.0-compute -x hlsl  -finclude-default-header -ast-dump  -o - %s | FileCheck %s

[numthreads(8,8,1)]
void CSMain(uint ID : SV_DispatchThreadID) {
// CHECK: FunctionDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-1]]:6 CSMain 'void (uint)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:18 ID 'uint'
// CHECK-NEXT: HLSLSV_DispatchThreadIDAttr
}
[numthreads(8,8,1)]
void CSMain1(uint2 ID : SV_DispatchThreadID) {
// CHECK: FunctionDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-1]]:6 CSMain1 'void (uint2)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:20 ID 'uint2'
// CHECK-NEXT: HLSLSV_DispatchThreadIDAttr
}
[numthreads(8,8,1)]
void CSMain2(uint3 ID : SV_DispatchThreadID) {
// CHECK: FunctionDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-1]]:6 CSMain2 'void (uint3)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:20 ID 'uint3'
// CHECK-NEXT: HLSLSV_DispatchThreadIDAttr
}
[numthreads(8,8,1)]
void CSMain3(uint3 : SV_DispatchThreadID) {
// CHECK: FunctionDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> line:[[@LINE-1]]:6 CSMain3 'void (uint3)'
// CHECK-NEXT: ParmVarDecl 0x{{[0-9a-fA-F]+}} <{{.*}}> col:20 'uint3'
// CHECK-NEXT: HLSLSV_DispatchThreadIDAttr
}
