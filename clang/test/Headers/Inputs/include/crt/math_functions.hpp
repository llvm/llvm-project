// required for __clang_cuda_runtime_wrapper.h tests
#pragma once
__device__ int                    __isinff(float);
__device__ int                    __isinf(double);
__device__ int                    __finitef(float);
__device__ int                    __isfinited(double);
__device__ int                    __isnanf(float);
__device__ int                    __isnan(double);
__device__ int                    __signbitf(float);
__device__ int                    __signbitd(double);
__device__ double                 max(double, double);
__device__ float                  max(float, float);
