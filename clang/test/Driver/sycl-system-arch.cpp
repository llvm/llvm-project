// Needs chmod
// UNSUPPORTED: system-windows
// XFAIL: target={{.*}}-zos{{.*}}

// RUN: mkdir -p %t
// RUN: cp %S/Inputs/offload-arch/offload_arch_sm_70_gfx906_gfx90a_xe_pvc %t/
// RUN: chmod +x %t/offload_arch_sm_70_gfx906_gfx90a_xe_pvc

/// For SYCL offloading, --offload-arch=native skips NVIDIA and AMD GPUs,
/// including an AMD GPU with target features.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=native --offload-arch-tool=%t/offload_arch_sm_70_gfx906_gfx90a_xe_pvc \
// RUN:   -c %s 2>&1 | FileCheck %s
// CHECK-NOT: "device-sycl ({{amdgpu|nvptx}}
// CHECK: offload, "device-sycl (spirv64-unknown-unknown:xe-pvc)"
// CHECK-NOT: "device-sycl ({{amdgpu|nvptx}}
