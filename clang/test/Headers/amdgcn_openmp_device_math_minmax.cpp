// RUN: %clang_cc1 -internal-isystem %S/Inputs/include -x c++ -fopenmp -triple x86_64-unknown-unknown -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm-bc %s -o %t-host.bc
// RUN: %clang_cc1 -internal-isystem %S/../../lib/Headers/openmp_wrappers -include __clang_openmp_device_functions.h -internal-isystem %S/../../lib/Headers/openmp_wrappers -internal-isystem %S/Inputs/include -x c++ -fopenmp -triple amdgpu-amd-amdhsa -aux-triple x86_64-unknown-unknown -fopenmp-targets=amdgpu-amd-amdhsa -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-host.bc -verify -o -

// The OpenMP math wrapper pulls in __clang_hip_math.h, whose ::min and ::max
// are a CUDA compatibility extension for HIP. In an OpenMP translation unit
// these calls are ordinary unqualified C++ lookups and must resolve to std.

// expected-no-diagnostics

#include <algorithm>
#include <cmath>

using namespace std;

long test_min(long __a, long __b) { return min(__a, __b); }

long test_max(long __a, long __b) { return max(__a, __b); }
