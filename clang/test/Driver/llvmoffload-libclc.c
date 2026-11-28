// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx908 \
// RUN:   -foffload-via-llvm \
// RUN:   -resource-dir %S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   -x hip %s 2>&1 | FileCheck %s --check-prefix=HIP
// HIP: "-cc1" "-triple" "amdgpu9.08-amd-amdhsa-llvm"
// HIP-SAME: "-mlink-builtin-bitcode"
// HIP-SAME: "{{.*}}resource_dir_with_per_target_subdir{{/|\\\\}}lib{{/|\\\\}}amdgpu-amd-amdhsa{{/|\\\\}}libclc.bc"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=gfx908 \
// RUN:   -foffload-via-llvm --no-offloadlib \
// RUN:   -resource-dir %S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   -x hip %s 2>&1 | FileCheck %s --check-prefix=HIP-NO-OFFLOADLIB
// HIP-NO-OFFLOADLIB: "-cc1" "-triple" "amdgpu9.08-amd-amdhsa-llvm"
// HIP-NO-OFFLOADLIB-NOT: "-mlink-builtin-bitcode"
// HIP-NO-OFFLOADLIB-NOT: libclc.bc

// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=sm_52 \
// RUN:   -foffload-via-llvm \
// RUN:   -resource-dir %S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda \
// RUN:   -x cuda %s 2>&1 | FileCheck %s --check-prefix=CUDA
// CUDA: "-cc1" "-triple" "nvptx64-nvidia-cuda-llvm"
// CUDA-SAME: "-mlink-builtin-bitcode"
// CUDA-SAME: "{{.*}}resource_dir_with_per_target_subdir{{/|\\\\}}lib{{/|\\\\}}nvptx64-nvidia-cuda{{/|\\\\}}libclc.bc"

// RUN: %clang -### --target=x86_64-unknown-linux-gnu --offload-arch=sm_52 \
// RUN:   -foffload-via-llvm --no-offloadlib \
// RUN:   -resource-dir %S/Inputs/resource_dir_with_per_target_subdir \
// RUN:   --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda \
// RUN:   -x cuda %s 2>&1 | FileCheck %s --check-prefix=CUDA-NO-OFFLOADLIB
// CUDA-NO-OFFLOADLIB: "-cc1" "-triple" "nvptx64-nvidia-cuda-llvm"
// CUDA-NO-OFFLOADLIB-NOT: "-mlink-builtin-bitcode"
// CUDA-NO-OFFLOADLIB-NOT: libclc.bc

void f(void) {}
