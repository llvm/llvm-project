// Mimics the part of NVIDIA's curand_mtgp32_kernel.h that matters here: in host
// mode it redeclares blockDim/threadIdx using the runtime's dim3/uint3 types.
#pragma once
#ifndef __CUDA_ARCH__
extern const dim3 blockDim;
extern const uint3 threadIdx;
#endif
