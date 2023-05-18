// RUN: %clang_cc1 %s --std=c++11 -triple x86_64-unknown-linux -emit-llvm -o - -debug-info-kind=limited -dwarf-version=2 -debugger-tuning=gdb | FileCheck %s

#include "Inputs/cuda.h"

__device__ void f();
template<void(*F)()> __global__ void t() { F(); }
__host__ void g() { t<f><<<1,1>>>(); }

// Ensure the value of device-function (as value template parameter) is null.
// CHECK: !DITemplateValueParameter(name: "F", type: !{{[0-9]+}}, value: null)
