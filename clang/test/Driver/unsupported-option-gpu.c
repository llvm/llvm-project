/// Some target-specific options are ignored for GPU, so %clang exits with code 0.
// DEFINE: %{check} = %clang -### -c -mcmodel=medium

// RUN: %{check} -x cuda %s --cuda-path=%S/Inputs/CUDA/usr/local/cuda --offload-arch=sm_60 --no-cuda-version-check -fbasic-block-sections=all
// RUN: %{check} -x hip %s --rocm-path=%S/Inputs/rocm -nogpulib -nogpuinc
