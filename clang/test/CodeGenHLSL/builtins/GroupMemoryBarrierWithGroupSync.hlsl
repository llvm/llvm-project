// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.0-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -DTARGET=dx -DFNATTRS=noundef -check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -DTARGET=spv -DFNATTRS="spir_func noundef" -check-prefixes=CHECK,CHECK-SPIRV

// CHECK-DXIL: define void @
// CHECK-SPIRV: define spir_func void @
void test_GroupMemoryBarrierWithGroupSync() {
// CHECK: call void @llvm.[[TARGET]].groupMemoryBarrierWithGroupSync()
  GroupMemoryBarrierWithGroupSync();
}

// CHECK: declare void @llvm.[[TARGET]].groupMemoryBarrierWithGroupSync() #[[ATTRS:[0-9]+]]
// CHECK-NOT: attributes #[[ATTRS]] = {{.+}}memory(none){{.+}}
// CHECK: attributes #[[ATTRS]] = {
