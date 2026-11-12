// HIP on AMDGPU.
// RUN: %clang_cc1 -std=c++17 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple amdgpu9.0a-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -x hip -fcuda-is-device -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_builtin_vars.h
// RUN: %clang_cc1 -std=c++20 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple amdgpu9.0a-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -x hip -fcuda-is-device -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_builtin_vars.h

// HIP on SPIR-V.
// RUN: %clang_cc1 -std=c++20 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple spirv64-amd-amdhsa -aux-triple x86_64-unknown-unknown \
// RUN:   -x hip -fcuda-is-device -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_builtin_vars.h

// CUDA on NVPTX.
// RUN: %clang_cc1 -std=c++17 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-unknown \
// RUN:   -x cuda -fcuda-is-device -target-cpu sm_70 -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_builtin_vars.h
// RUN: %clang_cc1 -std=c++20 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple nvptx64-nvidia-cuda -aux-triple x86_64-unknown-unknown \
// RUN:   -x cuda -fcuda-is-device -target-cpu sm_70 -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_builtin_vars.h

// HIP host compilation.
// RUN: %clang_cc1 -std=c++20 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple x86_64-unknown-unknown -aux-triple amdgpu9.0a-amd-amdhsa \
// RUN:   -x hip -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_builtin_vars.h

// CUDA host compilation.
// RUN: %clang_cc1 -std=c++20 -internal-isystem %S/Inputs/include \
// RUN:   -internal-isystem %S/../../lib/Headers \
// RUN:   -triple x86_64-unknown-unknown -aux-triple nvptx64-nvidia-cuda \
// RUN:   -aux-target-cpu sm_70 -x cuda -fsyntax-only -verify %s \
// RUN:   -include __clang_gpu_builtin_vars.h

// expected-no-diagnostics

__attribute__((global)) void test_kernel(unsigned *out) {
  unsigned i = threadIdx.x + threadIdx.y + threadIdx.z;
  i += blockIdx.x + blockIdx.y + blockIdx.z;
  i += blockDim.x + blockDim.y + blockDim.z;
  i += gridDim.x + gridDim.y + gridDim.z;
  *out = i;
}
