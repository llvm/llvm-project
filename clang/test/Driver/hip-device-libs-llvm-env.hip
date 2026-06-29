// Do not use device library with llvm environment
// RUN: %clang -### --target=x86_64-linux-gnu -nogpuinc \
// RUN:   --offload-targets=amdgcn-amd-amdhsa-llvm --offload-arch=gfx90a \
// RUN:   -resource-dir=%S/Inputs/rocm_resource_dir \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefix=LLVMENV %s

// LLVMENV-NOT: -mlink-builtin-bitcode
// LLVMENV-NOT: oclc
// LLVMENV-NOT: ocml
// LLVMENV-NOT: ockl

// RUN: %clang -### --target=x86_64-linux-gnu --rocm-path=%S/Inputs/rocm \
// RUN:   --offload-targets=amdgcn-amd-amdhsa-llvm --offload-arch=gfx90a \
// RUN:   -resource-dir=%S/Inputs/rocm_resource_dir \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefix=INC %s

// INC: "-cc1" "-triple" "amdgcn-amd-amdhsa-llvm"
// INC-SAME: "-include" "__clang_gpu_device_functions.h"

// RUN: %clang -### --target=x86_64-linux-gnu -nogpuinc --rocm-path=%S/Inputs/rocm \
// RUN:   --offload-targets=amdgcn-amd-amdhsa-llvm --offload-arch=gfx90a \
// RUN:   -resource-dir=%S/Inputs/rocm_resource_dir \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefix=NOINC %s

// RUN: %clang -### --target=x86_64-linux-gnu -nohipwrapperinc --rocm-path=%S/Inputs/rocm \
// RUN:   --offload-targets=amdgcn-amd-amdhsa-llvm --offload-arch=gfx90a \
// RUN:   -resource-dir=%S/Inputs/rocm_resource_dir \
// RUN:   %S/Inputs/hip_multiple_inputs/b.hip \
// RUN: 2>&1 | FileCheck --check-prefix=NOINC %s

// NOINC-NOT: __clang_gpu_device_functions.h
