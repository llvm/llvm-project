// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-compute %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -DTARGET=dx -DFNATTRS=noundef
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -DTARGET=spv -DFNATTRS="spir_func noundef"

// CHECK: define [[FNATTRS]] i32 @
[numthreads(1, 1, 1)]
void main() {
  while (true) {
// CHECK: call void @llvm.[[TARGET]].groupMemoryBarrierWithGroupSync()
  GroupMemoryBarrierWithGroupSync();
  break;
  }
}

// CHECK: declare void @llvm.[[TARGET]].groupMemoryBarrierWithGroupSync() #[[ATTRS:[0-9]+]]
// CHECK-NOT: attributes #[[ATTRS]] = {{.+}}memory(none){{.+}}
// CHECK: attributes #[[ATTRS]] = {
