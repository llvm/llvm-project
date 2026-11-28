// The Clang pipeline doesn't run CallConvLowering for HLSL on SPIR-V, so a bool
// argument isn't extended, matching classic codegen.

// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -finclude-default-header -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s
// RUN: %clang_cc1 -x hlsl -triple spirv-unknown-vulkan-library -finclude-default-header -fclangir -emit-llvm %s -o - | FileCheck %s

// CHECK: define {{.*}}void @_Z9take_boolb(i1 noundef %{{[^,)]+}})
export void take_bool(bool b) {}
