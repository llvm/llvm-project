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
// RUN: | FileCheck -check-prefix=HIP-DEVICE-PGO %s
// HIP-DEVICE-PGO: "{{.*}}lld" {{.*}}"{{.*}}lib{{/|\\\\}}amdgcn-amd-amdhsa{{/|\\\\}}libclang_rt.profile.a"
