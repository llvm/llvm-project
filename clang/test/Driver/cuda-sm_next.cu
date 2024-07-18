// Tests CUDA compilation targeting sm_custom

// CC1 options level check.
// Check that by default we only supply sm_custom requires explicitly 
// overriding SM/PTX versions.
// RUN: not %clang -### -c --target=x86_64-linux-gnu --cuda-device-only  \
// RUN:    --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda %s \
// RUN:    --cuda-gpu-arch=sm_custom  2>&1 \
// RUN:   | FileCheck -check-prefixes=ERROR %s
//
// Check propagation of explicitly set sm and PTX versions to the tools.
// RUN: %clang -### -c --target=x86_64-linux-gnu \
// RUN:    --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda %s \
// RUN:    --cuda-gpu-arch=sm_custom --cuda-custom-sm=sm_111 --cuda-custom-ptx=222  2>&1 \
// RUN:   | FileCheck -check-prefixes=ARGS %s

// Preprocessor level checks.
// RUN: %clang -dD -E --target=x86_64-linux-gnu --cuda-device-only -nocudainc \
// RUN:    --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda %s \
// RUN:    --cuda-gpu-arch=sm_custom --cuda-custom-sm=sm_111 --cuda-custom-ptx=222  2>&1 \
// RUN:   | FileCheck -check-prefixes=PP %s

// PTX level checks. 
// RUN: %clang -S --target=x86_64-linux-gnu --cuda-device-only -nocudainc -nocudalib \
// RUN:      --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda -o - %s \
// RUN:      --cuda-gpu-arch=sm_custom --cuda-custom-sm=sm_111 --cuda-custom-ptx=222  2>&1 \
// RUN:   | FileCheck -check-prefixes=PTX %s


// ERROR: clang: error: offload target sm_custom requires both --cuda-custom_sm and --cuda_custom_ptx to be specified

// ARGS: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// We need to pass specific SM version to CC1, so that preprocessor can set __CUDA_ARCH__ macro
// and both SM and PTX to LLVM so it would generate correct PTX directives.
// ARGS-SAME: "--cuda-custom-sm=sm_111" "-mllvm" "--nvptx-custom-sm=sm_111" "-mllvm" "--nvptx-custom-ptx=222"
// ARGS-SAME: "-target-cpu" "sm_custom"
// ARGS-SAME: "-target-feature" "+ptx71"
// ARGS-NEXT: ptxas
// ARGS-SAME: "--gpu-name" "sm_111"
// ARGS-NEXT: fatbinary
// ARGS-SAME: "--image=profile=sm_111,file= 
// ARGS-SAME: "--image=profile=compute_111,file
//
//
// PP:  #define __NVPTX__ 1
// PP: #define __CUDA_ARCH__  1110
//
// PTX:  .version 22.2
// PTX:  .target sm_111
