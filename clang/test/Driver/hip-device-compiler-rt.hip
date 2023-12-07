// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Check device compiler-rt is linked when available.

// RUN: %clang -### --target=x86_64-linux-gnu --offload-arch=gfx1010 \
// RUN:   -nogpuinc -nostdinc --offload-device-only --rocm-path=%S/Inputs/rocm \
// RUN:   -resource-dir=%S/Inputs/device_compiler_rt_resource_dir \
// RUN:   %s 2>&1 | FileCheck %s

// CHECK: "-mlink-bitcode-file" "{{[^"]+(/|\\\\)device_compiler_rt_resource_dir(/|\\\\)lib(64)?(/|\\\\)amdgcn-amd-amdhsa(/|\\\\).*}}libclang_rt.builtins.a"
// CHECK-SAME: "-mlink-builtin-bitcode" "[[DEVICELIB_DIR:[^"]+(/|\\\\)rocm(/|\\\\)amdgcn(/|\\\\).*]]hip.bc"
// CHECK-SAME: "-mlink-builtin-bitcode" "[[DEVICELIB_DIR]]ocml.bc"
// CHECK-SAME: "-mlink-builtin-bitcode" "[[DEVICELIB_DIR]]ockl.bc"
// CHECK-SAME: "-mlink-builtin-bitcode" "[[DEVICELIB_DIR]]oclc_isa_version_{{[0-9]+}}.bc"
