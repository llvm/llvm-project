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


// COM: [[LLVM_LINK:"*.llvm-link"]]
// COM-SAME: "{{.*}}hip.amdgcn.bc" "{{.*}}opencl.amdgcn.bc"
// COM-SAME: "{{.*}}ocml.amdgcn.bc" "{{.*}}ockl.amdgcn.bc"
// FLUSHD-SAME: {{.*}} "{{.*}}oclc_daz_opt_on.amdgcn.bc"
// NOFLUSHD-SAME: {{.*}} "{{.*}}oclc_daz_opt_off.amdgcn.bc"
// COM-SAME: {{.*}} "-o" "{{.*}}-gfx900-linked-{{.*bc}}"

