// From CUDA 13.0 on, the CUDA C++ Core Libraries are in include/cccl of the
// CUDA installation. Check that it is a system include directory, right after
// include/, for both the device and the host compilation.

// RUN: %clang -### --target=x86_64-unknown-linux-gnu --cuda-gpu-arch=sm_75 \
// RUN:   --cuda-path=%S/Inputs/CUDA_130/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=CCCL

// CCCL: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// CCCL-SAME: "-internal-isystem" "[[CUDA:[^"]*Inputs/CUDA_130/usr/local/cuda]]{{[/\\]+}}include" "-internal-isystem" "[[CUDA]]/include{{[/\\]+}}cccl"
// CCCL: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// CCCL-SAME: "-internal-isystem" "[[CUDA]]{{[/\\]+}}include" "-internal-isystem" "[[CUDA]]{{[/\\]+}}include{{[/\\]+}}cccl"

// Older CUDA installations have no include/cccl.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu --cuda-gpu-arch=sm_75 \
// RUN:   --cuda-path=%S/Inputs/CUDA_111/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOCCCL

// -nogpuinc leaves out all of the CUDA installation's include directories.
// RUN: %clang -### --target=x86_64-unknown-linux-gnu --cuda-gpu-arch=sm_75 \
// RUN:   -nogpuinc --cuda-path=%S/Inputs/CUDA_130/usr/local/cuda %s 2>&1 \
// RUN:   | FileCheck %s --check-prefix=NOCCCL

// NOCCCL: "-cc1" "-triple" "nvptx64-nvidia-cuda"
// NOCCCL-NOT: include{{[/\\]+}}cccl"
// NOCCCL: "-cc1" "-triple" "x86_64-unknown-linux-gnu"
// NOCCCL-NOT: include{{[/\\]+}}cccl"
