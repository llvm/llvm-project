// RUN: %clang_cc1 -fsyntax-only -triple x86_64 -aux-triple amdgcn -verify %s
// expected-no-diagnostics
#include "Inputs/cuda.h"

__device__ void f(_Float16 x);

__device__ _Float16 x = 1.0f16;
