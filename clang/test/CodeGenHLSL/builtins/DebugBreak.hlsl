// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.10-library \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s -check-prefixes=CHECK

// CHECK: define hidden void @_Z15test_DebugBreakv()
// CHECK: call void @_ZN4hlsl10DebugBreakEv()
// CHECK: call void @_ZN4hlsl10DebugBreakEv()
// CHECK: ret void

// CHECK: define linkonce_odr hidden void @_ZN4hlsl10DebugBreakEv()
// CHECK: call void @llvm.debugtrap()
// CHECK: ret void

using namespace hlsl;

void test_DebugBreak() {
  DebugBreak();
  hlsl::DebugBreak();
}
