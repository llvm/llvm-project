// REQUIRES: x86-registered-target, amdgpu-registered-target
// UNSUPPORTED: system-windows

// Build a fake resource dir containing host and device profile runtimes.
// RUN: rm -rf %t && mkdir -p %t/lib/x86_64-unknown-linux %t/lib/amdgcn-amd-amdhsa
// RUN: touch %t/lib/x86_64-unknown-linux/libclang_rt.profile.a
// RUN: touch %t/lib/x86_64-unknown-linux/libclang_rt.profile_rocm.a
// RUN: touch %t/lib/amdgcn-amd-amdhsa/libclang_rt.profile.a
// RUN: touch %t.o

// HIP host link with PGO links clang_rt.profile_rocm.
// RUN: %clang -### --hip-link --target=x86_64-unknown-linux \
// RUN:   -fprofile-instr-generate -resource-dir=%t \
// RUN:   --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=HIP-PGO %s
// profile_rocm must precede the base profile so the base archive stays inert.
// HIP-PGO: "{{.*}}libclang_rt.profile_rocm.a"
// The constructor-only hipModuleLoad* interceptor object is force-linked.
// HIP-PGO: "-u" "__llvm_profile_offload_register_dynamic_module"
// HIP-PGO: "{{.*}}libclang_rt.profile.a"

// Without profiling, the ROCm device-profile runtime is not linked.
// RUN: %clang -### --hip-link --target=x86_64-unknown-linux \
// RUN:   -resource-dir=%t --rocm-path=%S/Inputs/rocm %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=HIP-NOPGO %s
// HIP-NOPGO-NOT: libclang_rt.profile_rocm.a

// A non-HIP host link with PGO does not link the ROCm device-profile runtime.
// RUN: %clang -### --target=x86_64-unknown-linux \
// RUN:   -fprofile-instr-generate -resource-dir=%t %t.o 2>&1 \
// RUN:   | FileCheck -check-prefix=HOST-PGO %s
// HOST-PGO: "{{.*}}libclang_rt.profile.a"
// HOST-PGO-NOT: libclang_rt.profile_rocm.a

// HIP device-only link with PGO links the device profile runtime.
// RUN: %clang -### -x hip --offload-device-only --target=x86_64-unknown-linux \
// RUN: --offload-arch=gfx90a -fprofile-generate -resource-dir=%t \
// RUN: -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefix=HIP-DEVICE-PGO --implicit-check-not=-plugin-opt=-amdgpu-internalize-symbols %s
// HIP-DEVICE-PGO: "-cc1"
// HIP-DEVICE-PGO-SAME: "-emit-obj"
// HIP-DEVICE-PGO-SAME: "-mllvm" "-amdgpu-internalize-symbols"
// HIP-DEVICE-PGO: "{{.*}}lld"
// HIP-DEVICE-PGO-SAME: "{{.*}}lib{{/|\\\\}}amdgcn-amd-amdhsa{{/|\\\\}}libclang_rt.profile.a"

// An RDC non-LTO link also keeps bitcode library definitions external.
// RUN: %clang -x c --target=x86_64-unknown-linux-gnu -c %s -o %t.host.o
// RUN: %clang -x hip --offload-device-only --target=x86_64-unknown-linux \
// RUN: --offload-arch=gfx90a -fgpu-rdc -fno-offload-lto -nogpuinc -nogpulib \
// RUN: -c %s -o %t.device.o
// RUN: clang-offload-bundler -type=o \
// RUN: -targets=host-x86_64-unknown-linux-gnu,hip-amdgpu9.0a-amd-amdhsa--gfx90a \
// RUN: -input=%t.host.o -input=%t.device.o -output=%t.rdc.o
// RUN: %clang -### --hip-link --target=x86_64-unknown-linux \
// RUN: --offload-arch=gfx90a -fgpu-rdc -fno-offload-lto -fprofile-generate \
// RUN: -resource-dir=%t --rocm-path=%S/Inputs/rocm -nogpuinc -nogpulib \
// RUN: %t.rdc.o 2>&1 \
// RUN: | FileCheck -check-prefix=HIP-RDC-PGO --implicit-check-not=-plugin-opt=-amdgpu-internalize-symbols %s
// HIP-RDC-PGO: clang-linker-wrapper
// HIP-RDC-PGO-SAME: "--device-compiler=amdgpu-amd-amdhsa=-fprofile-generate"

// Full offload LTO still enables AMDGPU internalization.
// RUN: %clang -### -x hip --offload-device-only --target=x86_64-unknown-linux \
// RUN: --offload-arch=gfx90a -fprofile-generate -foffload-lto=full \
// RUN: -resource-dir=%t -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefix=HIP-DEVICE-LTO-PGO %s
// HIP-DEVICE-LTO-PGO: "-cc1"
// HIP-DEVICE-LTO-PGO-SAME: "-emit-llvm-bc"
// HIP-DEVICE-LTO-PGO: "{{.*}}lld"
// HIP-DEVICE-LTO-PGO-SAME: "-plugin-opt=-amdgpu-internalize-symbols"
// HIP-DEVICE-LTO-PGO-SAME: "{{.*}}lib{{/|\\\\}}amdgcn-amd-amdhsa{{/|\\\\}}libclang_rt.profile.a"

// A normal non-profile device link also internalizes the main module in cc1,
// not while linking the resulting native object.
// RUN: %clang -### -x hip --offload-device-only --target=x86_64-unknown-linux \
// RUN: --offload-arch=gfx90a -resource-dir=%t -nogpuinc -nogpulib %s 2>&1 \
// RUN: | FileCheck -check-prefix=HIP-DEVICE-NOPGO --implicit-check-not=-plugin-opt=-amdgpu-internalize-symbols %s
// HIP-DEVICE-NOPGO: "-cc1"
// HIP-DEVICE-NOPGO-SAME: "-emit-obj"
// HIP-DEVICE-NOPGO-SAME: "-mllvm" "-amdgpu-internalize-symbols"
// HIP-DEVICE-NOPGO: "{{.*}}lld"
// HIP-DEVICE-NOPGO-NOT: libclang_rt.profile.a
