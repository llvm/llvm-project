// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only \
// RUN:   -isystem %S/Inputs -verify %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:   -isystem %S/Inputs -fcuda-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only \
// RUN:   -isystem %S/Inputs -verify=redecl -Wnvcc-compat %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:   -isystem %S/Inputs -fcuda-is-device -Wnvcc-compat -verify=redecl %s

// expected-no-diagnostics
#include "cuda.h"

__device__ void f(); // redecl-note {{previous declaration is here}}

void f() {} // redecl-warning {{target-attribute based function overloads are not supported by NVCC and will be treated as a function redeclaration:new declaration is __host__ function, old declaration is __device__ function}}

void g(); // redecl-note {{previous declaration is here}}

__device__ void g() {} // redecl-warning {{target-attribute based function overloads are not supported by NVCC and will be treated as a function redeclaration:new declaration is __device__ function, old declaration is __host__ function}}
