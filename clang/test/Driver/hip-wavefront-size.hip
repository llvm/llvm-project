// REQUIRES: amdgpu-registered-target

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm --cuda-device-only %s \
// RUN:   2>&1 | FileCheck %s --check-prefixes=WAVE64
// WAVE64: "-mlink-builtin-bitcode" "{{.*}}oclc_wavefrontsize64_on.bc"{{.*}} "-target-cpu" "gfx900"

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx1010 \
// RUN:   --rocm-path=%S/Inputs/rocm --cuda-device-only %s \
// RUN:   2>&1 | FileCheck %s --check-prefixes=WAVE32
// WAVE32: "-mlink-builtin-bitcode" "{{.*}}oclc_wavefrontsize64_off.bc"{{.*}} "-target-cpu" "gfx1010"

// RUN: %clang -### -target x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx1010 \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm --cuda-device-only %s \
// RUN:   2>&1 | FileCheck %s --check-prefixes=BOTH
// BOTH-DAG: "-mlink-builtin-bitcode" "{{.*}}oclc_wavefrontsize64_on.bc"{{.*}} "-target-cpu" "gfx900"
// BOTH-DAG: "-mlink-builtin-bitcode" "{{.*}}oclc_wavefrontsize64_off.bc"{{.*}} "-target-cpu" "gfx1010"
