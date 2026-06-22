// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   dxil-pc-shadermodel6.3-library %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -DTARGET=dx -check-prefixes=CHECK,CHECK-DXIL
// RUN: %clang_cc1 -finclude-default-header -x hlsl -triple \
// RUN:   spirv-unknown-vulkan-compute %s \
// RUN:   -emit-llvm -disable-llvm-passes -o - | FileCheck %s \
// RUN:   -DTARGET=spv -check-prefixes=CHECK,CHECK-SPIRV

// CHECK-DXIL: define hidden void @
// CHECK-SPIRV: define hidden spir_func void @
void test_DeviceMemoryBarrier() {
// CHECK-DXIL: call void @llvm.[[TARGET]].device.memory.barrier()
// CHECK-SPIRV: call void @llvm.[[TARGET]].device.memory.barrier()
  DeviceMemoryBarrier();
}

// CHECK: declare void @llvm.[[TARGET]].device.memory.barrier() #[[ATTRS:[0-9]+]]
// CHECK-NOT: attributes #[[ATTRS]] = {{.+}}memory(none){{.+}}
// CHECK: attributes #[[ATTRS]] = {{.+}}convergent{{.+}}
