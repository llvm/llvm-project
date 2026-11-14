// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.10-library \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s
// RUN: %clang_cc1 -finclude-default-header -triple spirv-pc-vulkan-library \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK: define hidden{{( spir_func | )}}void @_Z15test_DebugBreakv()
// CHECK: call{{( spir_func | )}}void @_ZN4hlsl10DebugBreakEv()
// CHECK: call{{( spir_func | )}}void @_ZN4hlsl10DebugBreakEv()
// CHECK: ret void

// CHECK: define linkonce_odr hidden{{( spir_func | )}}void @_ZN4hlsl10DebugBreakEv()
// CHECK: call void @llvm.debugtrap()
// CHECK: ret void

using namespace hlsl;

void test_DebugBreak() {
  DebugBreak();
  hlsl::DebugBreak();
}
