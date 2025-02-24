#include "../Inputs/cuda.h"

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir \
// RUN:            -x cuda -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-HOST --input-file=%t.cir %s

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir \
// RUN:            -fcuda-is-device -emit-cir -target-sdk-version=12.3 \
// RUN:            %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR-DEVICE --input-file=%t.cir %s

namespace ns {
    __global__ void cpp_global_function_1(int a, int* b, float c) {}
    // CIR-HOST: cir.func @_ZN2ns36__device_stub__cpp_global_function_1EiPif
    // CIR-DEVICE: cir.func @_ZN2ns21cpp_global_function_1EiPif

    __global__ void cpp_global_function_2(int a, int* b, float c) {}

    // CIR-HOST: cir.func @_ZN2ns36__device_stub__cpp_global_function_2EiPif
    // CIR-DEVICE: cir.func @_ZN2ns21cpp_global_function_2EiPif

    __host__ void cpp_host_function_1(int a, int* b, float c) {}

    // CIR-HOST: cir.func @_ZN2ns19cpp_host_function_1EiPif

    __host__ void cpp_host_function_2(int a, int* b, float c) {}

    // CIR-HOST: cir.func @_ZN2ns19cpp_host_function_2EiPif

    __device__ void cpp_device_function_1(int a, int* b, float c) {}

    // CIR-DEVICE: cir.func @_ZN2ns21cpp_device_function_1EiPif

    __device__ void cpp_device_function_2(int a, int* b, float c) {}

    // CIR-DEVICE: cir.func @_ZN2ns21cpp_device_function_2EiPif
}

__global__ void cpp_global_function_1(int a, int* b, float c) {}

// CIR-HOST: cir.func @_Z36__device_stub__cpp_global_function_1iPif
// CIR-DEVICE: cir.func @_Z21cpp_global_function_1iPif

__global__ void cpp_global_function_2(int a, int* b, float c) {}

// CIR-HOST: cir.func @_Z36__device_stub__cpp_global_function_2iPif
// CIR-DEVICE: cir.func @_Z21cpp_global_function_2iPif

__host__ void cpp_host_function_1(int a, int* b, float c) {}

// CIR-HOST: cir.func @_Z19cpp_host_function_1iPif

__host__ void cpp_host_function_2(int a, int* b, float c) {}

// CIR-HOST: cir.func @_Z19cpp_host_function_2iPif

__device__ void cpp_device_function_1(int a, int* b, float c) {}

// CIR-DEVICE: cir.func @_Z21cpp_device_function_1iPif

__device__ void cpp_device_function_2(int a, int* b, float c) {}

// CIR-DEVICE: cir.func @_Z21cpp_device_function_2iPif

extern "C" {
    __global__ void c_global_function_1(int a, int* b, float c) {}

    // CIR-HOST: cir.func @__device_stub__c_global_function_1
    // CIR-DEVICE: cir.func @c_global_function_1

    __global__ void c_global_function_2(int a, int* b, float c) {}

    // CIR-HOST: cir.func @__device_stub__c_global_function_2
    // CIR-DEVICE: cir.func @c_global_function_2

    __host__ void c_host_function_1(int a, int* b, float c) {}

    // CIR-HOST: cir.func @c_host_function_1

    __host__ void c_host_function_2(int a, int* b, float c) {}

    // CIR-HOST: cir.func @c_host_function_2

    __device__ void c_device_function_1(int a, int* b, float c) {}

    // CIR-DEVICE: cir.func @c_device_function_1

    __device__ void c_device_function_2(int a, int* b, float c) {}

    // CIR-DEVICE: cir.func @c_device_function_2
}