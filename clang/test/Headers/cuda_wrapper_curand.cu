// Check that __clang_cuda_runtime_wrapper.h does not require cuRAND's
// curand_mtgp32_kernel.h, which is not part of every CUDA installation (for
// example one made of only cuda-nvcc and cuda-cudart-dev), but still includes
// it with dim3/uint3 redirected to the builtin types when it does exist.

// REQUIRES: x86-registered-target

// Inputs/include does not provide curand_mtgp32_kernel.h, which models a CUDA
// installation without cuRAND. A TU that does not use cuRAND must compile.
// RUN: %clang -target x86_64-unknown-linux-gnu -x cuda --cuda-host-only \
// RUN:   -fsyntax-only -nocudalib \
// RUN:   --cuda-path=%S/../Driver/Inputs/CUDA/usr/local/cuda \
// RUN:   -isystem %S/Inputs/include -isystem %S/../../lib/Headers %s

// A TU that asks for the missing header must get the error at its own
// #include, not at a line inside the wrapper.
// RUN: not %clang -target x86_64-unknown-linux-gnu -x cuda --cuda-host-only \
// RUN:   -fsyntax-only -nocudalib \
// RUN:   --cuda-path=%S/../Driver/Inputs/CUDA/usr/local/cuda \
// RUN:   -isystem %S/Inputs/include -isystem %S/../../lib/Headers \
// RUN:   -DINCLUDE_CURAND %s 2>&1 | FileCheck %s --check-prefix=MISSING
// MISSING: {{.*}}cuda_wrapper_curand.cu:{{[0-9]+}}:{{[0-9]+}}: fatal error: 'curand_mtgp32_kernel.h' file not found

// With the header present, Inputs/curand redeclares blockDim and threadIdx
// the way NVIDIA's header does. That only compiles if the wrapper has already
// included it with dim3 and uint3 redirected to the builtin types.
// RUN: %clang -target x86_64-unknown-linux-gnu -x cuda --cuda-host-only \
// RUN:   -fsyntax-only -nocudalib \
// RUN:   --cuda-path=%S/../Driver/Inputs/CUDA/usr/local/cuda \
// RUN:   -isystem %S/Inputs/include -isystem %S/Inputs/curand \
// RUN:   -isystem %S/../../lib/Headers -DINCLUDE_CURAND %s

#ifdef INCLUDE_CURAND
#include <curand_mtgp32_kernel.h>
#endif

__global__ void k() {}
