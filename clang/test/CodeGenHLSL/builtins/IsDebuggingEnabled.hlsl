// RUN: %clang_cc1 -finclude-default-header -triple dxil-pc-shadermodel6.10-library \
// RUN:   -emit-llvm -disable-llvm-passes -o - %s | FileCheck %s

// CHECK-LABEL: define hidden noundef i1 @_Z4testv()
// CHECK: %[[RESULT:.*]] = call noundef i1 @_ZN4hlsl2dx18IsDebuggingEnabledEv()
// CHECK: ret i1 %[[RESULT]]

// CHECK-LABEL: define linkonce_odr hidden noundef i1 @_ZN4hlsl2dx18IsDebuggingEnabledEv()
// CHECK: %[[RESULT:.*]] = call i1 @llvm.dx.isdebuggingenabled()
// CHECK: ret i1 %[[RESULT]]

bool test() {
  return hlsl::dx::IsDebuggingEnabled();
}
