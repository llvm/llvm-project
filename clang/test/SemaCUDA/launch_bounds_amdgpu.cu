// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcuda-is-device \
// RUN:   -triple amdgpu9.0a-amd-amdhsa -verify %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fcuda-is-device \
// RUN:   -triple spirv64-amd-amdhsa -verify %s

// expected-no-diagnostics

#include "Inputs/cuda.h"

// The one- and two-argument forms are consumed by AMDGPU codegen.
__launch_bounds__(128) void Test1Arg(void);
__launch_bounds__(128, 2) void Test2Args(void);

// The third argument (maxclusterrank) is not yet handled on AMDGPU; it is
// silently ignored rather than triggering the NVPTX sm_90 diagnostic.
__launch_bounds__(128, 2, 4) void Test3Args(void);
