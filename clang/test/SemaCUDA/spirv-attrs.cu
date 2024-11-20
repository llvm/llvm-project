// expected-no-diagnostics

// RUN: %clang_cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu \
// RUN:   -fcuda-is-device -verify -fsyntax-only %s

#include "Inputs/cuda.h"

__attribute__((reqd_work_group_size(128, 1, 1)))
__global__ void reqd_work_group_size_128_1_1() {}

__attribute__((work_group_size_hint(2, 2, 2)))
__global__ void work_group_size_hint_2_2_2() {}

__attribute__((vec_type_hint(int)))
__global__ void vec_type_hint_int() {}

__attribute__((intel_reqd_sub_group_size(64)))
__global__ void intel_reqd_sub_group_size_64() {}
