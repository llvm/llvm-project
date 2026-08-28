// Host-side compilation on x86 (no errors expected).
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple nvptx64 -fopenmp -x c -fsyntax-only -verify=host %s

// Device-side compilation for targets without 128-bit float/complex support (no errors expected).
// FIXME: The current behavior disables diagnostic for devices unconditionally, but once a way to correctly check
// for 128-bit support is implemented, we should update this test to expect an error for targets that do not support 128-bit float/complex types.

// RUN: %clang_cc1 -triple nvptx64 -aux-triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-is-target-device -x c -emit-llvm %s -o -  | FileCheck %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -aux-triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-is-target-device -x c -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu -fopenmp -fopenmp-is-target-device -x c -emit-llvm %s -o - | FileCheck %s

// host-no-diagnostics
// device-no-diagnostics
typedef _Complex float __cfloat128 __attribute__ ((__mode__ (__TC__)));

// FIXME: OpenMP codege seems to be generating incorrectly a pair of 32-bit floats for the 128-bit complex float type.
// This needs to be updated once the OpenMP codegen is fixed to generate the correct type for 128-bit complex float.

//CHECK: @A = {{.*}}global { float, float } zeroinitializer, align 4
#pragma omp declare target
_Complex float A;
#pragma omp end declare target
