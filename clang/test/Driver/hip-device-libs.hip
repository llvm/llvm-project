// REQUIRES: clang-driver
// REQUIRES: x86-registered-target
// REQUIRES: amdgpu-registered-target

// Test flush-denormals-to-zero enabled uses oclc_daz_opt_on

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   -fcuda-flush-denormals-to-zero \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=COM,FLUSHD

// Test flush-denormals-to-zero disabled uses oclc_daz_opt_off

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   -x hip --cuda-gpu-arch=gfx900 \
// RUN:   --hip-device-lib-path=%S/Inputs/hip_dev_lib \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck %s --check-prefixes=COM,NOFLUSHD


// COM: {{"[^"]*clang[^"]*"}}
// COM-SAME: "-mlink-builtin-bitcode" "{{.*}}hip.amdgcn.bc"
// COM-SAME: "-mlink-builtin-bitcode" "{{.*}}opencl.amdgcn.bc"
// COM-SAME: "-mlink-builtin-bitcode" "{{.*}}ocml.amdgcn.bc"
// COM-SAME: "-mlink-builtin-bitcode" "{{.*}}ockl.amdgcn.bc"
// FLUSHD-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_daz_opt_on.amdgcn.bc"
// NOFLUSHD-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_daz_opt_off.amdgcn.bc"

