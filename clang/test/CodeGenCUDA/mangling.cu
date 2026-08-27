// REQUIRES: nvptx-registered-target

// RUN: %if cir-enabled %{ %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -x cuda -emit-cir -target-sdk-version=12.3 %s -o - | FileCheck --check-prefix=CIR-HOST %s %}
// RUN: %if cir-enabled %{ %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir -fcuda-is-device -emit-cir -target-sdk-version=12.3 %s -o - | FileCheck --check-prefix=CIR-DEVICE %s %}

// RUN: %if cir-enabled %{ %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -x cuda -emit-llvm -target-sdk-version=12.3 %s -o - | FileCheck --check-prefix=LLVM-HOST %s %}
// RUN: %if cir-enabled %{ %clang_cc1 -triple nvptx64-nvidia-cuda -fclangir -fcuda-is-device -emit-llvm -target-sdk-version=12.3 %s -o - | FileCheck --check-prefix=LLVM-DEVICE %s %}

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x cuda -emit-llvm -target-sdk-version=12.3 %s -o - | FileCheck --check-prefix=OGCG-HOST %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -emit-llvm -target-sdk-version=12.3 %s -o - | FileCheck --check-prefix=OGCG-DEVICE %s

#include "Inputs/cuda.h"

namespace ns {

// CIR-HOST:   cir.func {{.*}} @_ZN2ns36__device_stub__cpp_global_function_1EiPif
// CIR-DEVICE: cir.func {{.*}} @_ZN2ns21cpp_global_function_1EiPif
// LLVM-HOST:  define {{.*}} @_ZN2ns36__device_stub__cpp_global_function_1EiPif
// LLVM-DEVICE: define {{.*}} @_ZN2ns21cpp_global_function_1EiPif
// OGCG-HOST:  define {{.*}} @_ZN2ns36__device_stub__cpp_global_function_1EiPif
// OGCG-DEVICE: define {{.*}} @_ZN2ns21cpp_global_function_1EiPif
    __global__ void cpp_global_function_1(int a, int* b, float c) {}

// CIR-HOST:   cir.func {{.*}} @_ZN2ns36__device_stub__cpp_global_function_2EiPif
// CIR-DEVICE: cir.func {{.*}} @_ZN2ns21cpp_global_function_2EiPif
// LLVM-HOST:  define {{.*}} @_ZN2ns36__device_stub__cpp_global_function_2EiPif
// LLVM-DEVICE: define {{.*}} @_ZN2ns21cpp_global_function_2EiPif
// OGCG-HOST:  define {{.*}} @_ZN2ns36__device_stub__cpp_global_function_2EiPif
// OGCG-DEVICE: define {{.*}} @_ZN2ns21cpp_global_function_2EiPif
    __global__ void cpp_global_function_2(int a, int* b, float c) {}

// CIR-HOST:  cir.func {{.*}} @_ZN2ns19cpp_host_function_1EiPif
// LLVM-HOST: define {{.*}} @_ZN2ns19cpp_host_function_1EiPif
// OGCG-HOST: define {{.*}} @_ZN2ns19cpp_host_function_1EiPif
    __host__ void cpp_host_function_1(int a, int* b, float c) {}

// CIR-HOST:  cir.func {{.*}} @_ZN2ns19cpp_host_function_2EiPif
// LLVM-HOST: define {{.*}} @_ZN2ns19cpp_host_function_2EiPif
// OGCG-HOST: define {{.*}} @_ZN2ns19cpp_host_function_2EiPif
    __host__ void cpp_host_function_2(int a, int* b, float c) {}

// CIR-DEVICE:  cir.func {{.*}} @_ZN2ns21cpp_device_function_1EiPif
// LLVM-DEVICE: define {{.*}} @_ZN2ns21cpp_device_function_1EiPif
// OGCG-DEVICE: define {{.*}} @_ZN2ns21cpp_device_function_1EiPif
    __device__ void cpp_device_function_1(int a, int* b, float c) {}

// CIR-DEVICE:  cir.func {{.*}} @_ZN2ns21cpp_device_function_2EiPif
// LLVM-DEVICE: define {{.*}} @_ZN2ns21cpp_device_function_2EiPif
// OGCG-DEVICE: define {{.*}} @_ZN2ns21cpp_device_function_2EiPif
    __device__ void cpp_device_function_2(int a, int* b, float c) {}
} // namespace ns

// CIR-HOST:   cir.func {{.*}} @_Z36__device_stub__cpp_global_function_1iPif
// CIR-DEVICE: cir.func {{.*}} @_Z21cpp_global_function_1iPif
// LLVM-HOST:  define {{.*}} @_Z36__device_stub__cpp_global_function_1iPif
// LLVM-DEVICE: define {{.*}} @_Z21cpp_global_function_1iPif
// OGCG-HOST:  define {{.*}} @_Z36__device_stub__cpp_global_function_1iPif
// OGCG-DEVICE: define {{.*}} @_Z21cpp_global_function_1iPif
__global__ void cpp_global_function_1(int a, int* b, float c) {}

// CIR-HOST:   cir.func {{.*}} @_Z36__device_stub__cpp_global_function_2iPif
// CIR-DEVICE: cir.func {{.*}} @_Z21cpp_global_function_2iPif
// LLVM-HOST:  define {{.*}} @_Z36__device_stub__cpp_global_function_2iPif
// LLVM-DEVICE: define {{.*}} @_Z21cpp_global_function_2iPif
// OGCG-HOST:  define {{.*}} @_Z36__device_stub__cpp_global_function_2iPif
// OGCG-DEVICE: define {{.*}} @_Z21cpp_global_function_2iPif
__global__ void cpp_global_function_2(int a, int* b, float c) {}

// CIR-HOST:  cir.func {{.*}} @_Z19cpp_host_function_1iPif
// LLVM-HOST: define {{.*}} @_Z19cpp_host_function_1iPif
// OGCG-HOST: define {{.*}} @_Z19cpp_host_function_1iPif
__host__ void cpp_host_function_1(int a, int* b, float c) {}

// CIR-HOST:  cir.func {{.*}} @_Z19cpp_host_function_2iPif
// LLVM-HOST: define {{.*}} @_Z19cpp_host_function_2iPif
// OGCG-HOST: define {{.*}} @_Z19cpp_host_function_2iPif
__host__ void cpp_host_function_2(int a, int* b, float c) {}

// CIR-DEVICE:  cir.func {{.*}} @_Z21cpp_device_function_1iPif
// LLVM-DEVICE: define {{.*}} @_Z21cpp_device_function_1iPif
// OGCG-DEVICE: define {{.*}} @_Z21cpp_device_function_1iPif
__device__ void cpp_device_function_1(int a, int* b, float c) {}

// CIR-DEVICE:  cir.func {{.*}} @_Z21cpp_device_function_2iPif
// LLVM-DEVICE: define {{.*}} @_Z21cpp_device_function_2iPif
// OGCG-DEVICE: define {{.*}} @_Z21cpp_device_function_2iPif
__device__ void cpp_device_function_2(int a, int* b, float c) {}

extern "C" {

// CIR-HOST:   cir.func {{.*}} @__device_stub__c_global_function_1
// CIR-DEVICE: cir.func {{.*}} @c_global_function_1
// LLVM-HOST:  define {{.*}} @__device_stub__c_global_function_1
// LLVM-DEVICE: define {{.*}} @c_global_function_1
// OGCG-HOST:  define {{.*}} @__device_stub__c_global_function_1
// OGCG-DEVICE: define {{.*}} @c_global_function_1
    __global__ void c_global_function_1(int a, int* b, float c) {}

// CIR-HOST:   cir.func {{.*}} @__device_stub__c_global_function_2
// CIR-DEVICE: cir.func {{.*}} @c_global_function_2
// LLVM-HOST:  define {{.*}} @__device_stub__c_global_function_2
// LLVM-DEVICE: define {{.*}} @c_global_function_2
// OGCG-HOST:  define {{.*}} @__device_stub__c_global_function_2
// OGCG-DEVICE: define {{.*}} @c_global_function_2
    __global__ void c_global_function_2(int a, int* b, float c) {}

// CIR-HOST:  cir.func {{.*}} @c_host_function_1
// LLVM-HOST: define {{.*}} @c_host_function_1
// OGCG-HOST: define {{.*}} @c_host_function_1
    __host__ void c_host_function_1(int a, int* b, float c) {}

// CIR-HOST:  cir.func {{.*}} @c_host_function_2
// LLVM-HOST: define {{.*}} @c_host_function_2
// OGCG-HOST: define {{.*}} @c_host_function_2
    __host__ void c_host_function_2(int a, int* b, float c) {}

// CIR-DEVICE:  cir.func {{.*}} @c_device_function_1
// LLVM-DEVICE: define {{.*}} @c_device_function_1
// OGCG-DEVICE: define {{.*}} @c_device_function_1
    __device__ void c_device_function_1(int a, int* b, float c) {}

// CIR-DEVICE:  cir.func {{.*}} @c_device_function_2
// LLVM-DEVICE: define {{.*}} @c_device_function_2
// OGCG-DEVICE: define {{.*}} @c_device_function_2
    __device__ void c_device_function_2(int a, int* b, float c) {}
} // extern "C"
