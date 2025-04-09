// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -DTARGET=dx -DFNATTRS=noundef -check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -DTARGET=spv -DFNATTRS="spir_func noundef" -check-prefixes=CHECK,CHECK-SPIRV

// CHECK-DXIL: define void @
// CHECK-SPIRV: define spir_func void @
void test_GroupMemoryBarrierWithGroupSync() {
// CHECK-DXIL: call void @llvm.[[TARGET]].group.memory.barrier.with.group.sync()
// CHECK-SPIRV: call spir_func void @llvm.[[TARGET]].group.memory.barrier.with.group.sync()
  GroupMemoryBarrierWithGroupSync();
}

// CHECK: declare void @llvm.[[TARGET]].group.memory.barrier.with.group.sync() #[[ATTRS:[0-9]+]]
// CHECK-NOT: attributes #[[ATTRS]] = {{.+}}memory(none){{.+}}
// CHECK: attributes #[[ATTRS]] = {
