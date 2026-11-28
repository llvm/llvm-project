// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s
#include "Inputs/cuda.h"

__attribute__((amdgpu_inst_pref_pad_lines(2)))
__global__ void pad_two() {}

__attribute__((amdgpu_inst_pref_pad_lines(0)))
__global__ void pad_zero() {}

__attribute__((amdgpu_inst_pref_pad_lines(2)))
__device__ void not_a_kernel() {} // expected-error {{'amdgpu_inst_pref_pad_lines' attribute only applies to kernel functions}}
