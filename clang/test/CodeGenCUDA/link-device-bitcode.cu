// Test for linking with CUDA's libdevice as outlined in
// http://llvm.org/docs/NVPTXUsage.html#linking-with-libdevice
//
// REQUIRES: nvptx-registered-target
//
// Prepare bitcode file to link with
// RUN: %clang_cc1 -triple nvptx-unknown-cuda -emit-llvm-bc \
// RUN:    -disable-llvm-passes -o %t.bc %S/Inputs/device-code.ll
// RUN: %clang_cc1 -triple nvptx-unknown-cuda -emit-llvm-bc \
// RUN:    -disable-llvm-passes -o %t-2.bc %S/Inputs/device-code-2.ll
//
// Make sure function in device-code gets linked in and internalized.
// RUN: %clang_cc1 -triple nvptx-unknown-cuda -fcuda-is-device \
// RUN:    -mlink-cuda-bitcode %t.bc  -emit-llvm \
// RUN:    -disable-llvm-passes -o - %s \
// RUN:    | FileCheck %s -check-prefix CHECK-IR
//
// Make sure we can link two bitcode files.
// RUN: %clang_cc1 -triple nvptx-unknown-cuda -fcuda-is-device \
// RUN:    -mlink-cuda-bitcode %t.bc -mlink-cuda-bitcode %t-2.bc \
// RUN:    -emit-llvm -disable-llvm-passes -o - %s \
// RUN:    | FileCheck %s -check-prefix CHECK-IR -check-prefix CHECK-IR-2
//
// Make sure function in device-code gets linked but is not internalized
// without -fcuda-uses-libdevice
// RUN: %clang_cc1 -triple nvptx-unknown-cuda -fcuda-is-device \
// RUN:    -mlink-bitcode-file %t.bc -emit-llvm \
// RUN:    -disable-llvm-passes -o - %s \
// RUN:    | FileCheck %s -check-prefix CHECK-IR-NLD
//
// Make sure NVVMReflect pass is enabled in NVPTX back-end.
// RUN: %clang_cc1 -triple nvptx-unknown-cuda -fcuda-is-device \
// RUN:    -mlink-cuda-bitcode %t.bc -S -o /dev/null %s \
// RUN:    -mllvm -debug-pass=Structure 2>&1 \
// RUN:    | FileCheck %s -check-prefix CHECK-REFLECT

#include "Inputs/cuda.h"

__device__ float device_mul_or_add(float a, float b);
extern "C" __device__ double __nv_sin(double x);
extern "C" __device__ double __nv_exp(double x);

// CHECK-IR-LABEL: define void @_Z26should_not_be_internalizedPf(
// CHECK-PTX-LABEL: .visible .func _Z26should_not_be_internalizedPf(
__device__ void should_not_be_internalized(float *data) {}

// Make sure kernel call has not been internalized.
// CHECK-IR-LABEL: define void @_Z6kernelPfS_
// CHECK-PTX-LABEL: .visible .entry _Z6kernelPfS_(
__global__ __attribute__((used)) void kernel(float *out, float *in) {
  *out = device_mul_or_add(in[0], in[1]);
  *out += __nv_exp(__nv_sin(*out));
  should_not_be_internalized(out);
}

// Make sure device_mul_or_add() is present in IR, is internal and
// calls __nvvm_reflect().
// CHECK-IR-LABEL: define internal float @_Z17device_mul_or_addff(
// CHECK-IR-NLD-LABEL: define float @_Z17device_mul_or_addff(
// CHECK-IR: call i32 @__nvvm_reflect
// CHECK-IR: ret float

// Make sure we've linked in and internalized only needed functions
// from the second bitcode file.
// CHECK-IR-2-LABEL: define internal double @__nv_sin
// CHECK-IR-2-LABEL: define internal double @__nv_exp
// CHECK-IR-2-NOT: double @__unused

// Verify that NVVMReflect pass is among the passes run by NVPTX back-end.
// CHECK-REFLECT: Replace occurrences of __nvvm_reflect() calls with 0/1
