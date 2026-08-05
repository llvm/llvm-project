// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -aux-triple amdgcn -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -aux-triple spirv64-amd-amdhsa -verify %s
// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -aux-triple nvptx64 -verify %s
// expected-no-diagnostics
#include "Inputs/cuda.h"

__device__ void f(_Float16 x);

__device__ _Float16 x = 1.0f16;
