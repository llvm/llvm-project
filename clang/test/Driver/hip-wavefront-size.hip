// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm --cuda-device-only %s \
// RUN:   2>&1 | FileCheck %s --check-prefixes=WAVE64
// WAVE64: "-cc1" "-triple" "amdgpu9.00-amd-amdhsa"
// WAVE64-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_wavefrontsize64_on.bc"

// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx1010 \
// RUN:   --rocm-path=%S/Inputs/rocm --cuda-device-only %s \
// RUN:   2>&1 | FileCheck %s --check-prefixes=WAVE32
// WAVE32: "-cc1" "-triple" "amdgpu10.10-amd-amdhsa"
// WAVE32-SAME: "-mlink-builtin-bitcode" "{{.*}}oclc_wavefrontsize64_off.bc"

// RUN: %clang -### --target=x86_64-linux-gnu \
// RUN:   --cuda-gpu-arch=gfx1010 \
// RUN:   --cuda-gpu-arch=gfx900 \
// RUN:   --rocm-path=%S/Inputs/rocm --cuda-device-only %s \
// RUN:   2>&1 | FileCheck %s --check-prefixes=BOTH
// BOTH-DAG: "-triple" "amdgpu9.00-amd-amdhsa"{{.*}}"-mlink-builtin-bitcode" "{{.*}}oclc_wavefrontsize64_on.bc"
// BOTH-DAG: "-triple" "amdgpu10.10-amd-amdhsa"{{.*}}"-mlink-builtin-bitcode" "{{.*}}oclc_wavefrontsize64_off.bc"
