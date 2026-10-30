/// Needs chmod
// UNSUPPORTED: system-windows
// XFAIL: target={{.*}}-zos{{.*}}

// RUN: mkdir -p %t
// RUN: cp %S/Inputs/offload-arch/offload_arch_sm_70_gfx906_xe_pvc %t/
// RUN: chmod +x %t/offload_arch_sm_70_gfx906_xe_pvc

/// SYCL only supports Intel devices, so --offload-arch=native skips NVIDIA and
/// AMD GPUs.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=native --offload-arch-tool=%t/offload_arch_sm_70_gfx906_xe_pvc \
// RUN:   -c %s 2>&1 | FileCheck %s
// CHECK-NOT: {{sm_70|gfx906}}
// CHECK: offload, "device-sycl (spirv64-unknown-unknown:xe-pvc)"
// CHECK-NOT: {{sm_70|gfx906}}
