// Needs chmod
// UNSUPPORTED: system-windows
// XFAIL: target={{.*}}-zos{{.*}}

// RUN: mkdir -p %t
// RUN: cp %S/Inputs/offload-arch/offload_arch_sm_70_gfx906_gfx90a_xe_pvc %t/
// RUN: cp %S/Inputs/offload-arch/offload_arch_sm_70_gfx90a %t/
// RUN: chmod +x %t/offload_arch_sm_70_gfx906_gfx90a_xe_pvc
// RUN: chmod +x %t/offload_arch_sm_70_gfx90a

/// By default, clang waits at most 60 seconds for offload-arch, and on a slow
/// machine even starting the fake one can take longer. Set
/// CLANG_TOOLCHAIN_PROGRAM_TIMEOUT=0 so that clang waits for it without a
/// limit.

/// For SYCL offloading, --offload-arch=native asks offload-arch for Intel GPUs
/// only.
// RUN: env CLANG_TOOLCHAIN_PROGRAM_TIMEOUT=0 \
// RUN:   %clang -ccc-print-phases --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=native --offload-arch-tool=%t/offload_arch_sm_70_gfx906_gfx90a_xe_pvc \
// RUN:   -c %s 2>&1 | FileCheck %s
// CHECK-NOT: "device-sycl ({{amdgpu|nvptx}}
// CHECK: offload, "device-sycl (spirv64-unknown-unknown:xe-pvc)"
// CHECK-NOT: "device-sycl ({{amdgpu|nvptx}}

/// Without an Intel GPU, SYCL reports that no GPU was found, as HIP and CUDA
/// do.
// RUN: env CLANG_TOOLCHAIN_PROGRAM_TIMEOUT=0 \
// RUN:   not %clang -### --target=x86_64-unknown-linux-gnu -fsycl \
// RUN:   --offload-arch=native --offload-arch-tool=%t/offload_arch_sm_70_gfx90a \
// RUN:   -c %s 2>&1 | FileCheck -check-prefix=NO-GPU %s
// NO-GPU: error: cannot determine sycl architecture: No GPU detected in the system
