// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.6-compute %s -DSM66 -hlsl-entry foo \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s

// RUN: %clang_cc1 -std=hlsl2021 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.8-compute %s -hlsl-entry foo \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s --check-prefix=CHECK-SM68


// Make sure wavesize attribute get correct value for sm66 and sm68.
// CHECK:define void @foo()
// CHECK:"hlsl.wavesize"="8,0,0"

// CHECK-SM68:define void @foo()
// CHECK-SM68:"hlsl.wavesize"="8,128,64"

[numthreads(16,8,1)]
#ifdef SM66
[WaveSize(8)]
#else
[WaveSize(8, 128, 64)]
#endif
void foo() {

}
