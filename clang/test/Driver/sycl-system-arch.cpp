// Needs chmod
// UNSUPPORTED: system-windows
// On macOS CI, running the fake offload-arch can take longer than
// CLANG_TOOLCHAIN_PROGRAM_TIMEOUT, see
// https://github.com/llvm/llvm-project/issues/238903.
// UNSUPPORTED: system-darwin
// XFAIL: target={{.*}}-zos{{.*}}

// RUN: mkdir -p %t
// RUN: cp %S/Inputs/offload-arch/offload_arch_sm_70_gfx906_gfx90a_xe_pvc %t/
// RUN: cp %S/Inputs/offload-arch/offload_arch_sm_70_gfx90a %t/
// RUN: chmod +x %t/offload_arch_sm_70_gfx906_gfx90a_xe_pvc
// RUN: chmod +x %t/offload_arch_sm_70_gfx90a

/// For SYCL offloading, --offload-arch=native asks offload-arch for Intel GPUs
/// only.
// RUN: %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=native --offload-arch-tool=%t/offload_arch_sm_70_gfx906_gfx90a_xe_pvc \
// RUN:   -c %s 2>&1 | FileCheck %s
// CHECK-NOT: "device-sycl ({{amdgpu|nvptx}}
// CHECK: offload, "device-sycl (spirv64-unknown-unknown:xe-pvc)"
// CHECK-NOT: "device-sycl ({{amdgpu|nvptx}}

/// Without an Intel GPU, SYCL reports that no GPU was found, as HIP and CUDA
/// do.
// RUN: not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=native --offload-arch-tool=%t/offload_arch_sm_70_gfx90a \
// RUN:   -c %s 2>&1 | FileCheck -check-prefix=NO-GPU %s
// NO-GPU: error: cannot determine sycl architecture: No GPU detected in the system
