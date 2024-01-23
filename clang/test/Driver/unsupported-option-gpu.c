/// Some target-specific options are ignored for GPU, so %clang exits with code 0.
// DEFINE: %{gpu_opts} = --cuda-gpu-arch=sm_60 --cuda-path=%S/Inputs/CUDA/usr/local/cuda --no-cuda-version-check
// DEFINE: %{check} = %clang -### -c %{gpu_opts} -mcmodel=medium %s
// RUN: %{check} -fbasic-block-sections=all

// REDEFINE: %{gpu_opts} = -x hip --rocm-path=%S/Inputs/rocm -nogpulib
// RUN: %{check}
