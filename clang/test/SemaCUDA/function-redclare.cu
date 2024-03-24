// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only \
// RUN:   -isystem %S/Inputs -verify %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:   -isystem %S/Inputs -fcuda-is-device -verify %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fsyntax-only \
// RUN:   -isystem %S/Inputs -verify=redecl -Woffload-incompatible-redeclare %s
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fsyntax-only \
// RUN:   -isystem %S/Inputs -fcuda-is-device -Woffload-incompatible-redeclare -verify=redecl %s

// expected-no-diagnostics
#include "cuda.h"

__device__ void f(); // redecl-note {{previous declaration is here}}

void f() {} // redecl-warning {{incompatible host/device attribute with redeclaration: new declaration is __host__ function, old declaration is __device__ function. It will cause warning with nvcc}}

void g(); // redecl-note {{previous declaration is here}}

__device__ void g() {} // redecl-warning {{incompatible host/device attribute with redeclaration: new declaration is __device__ function, old declaration is __host__ function. It will cause warning with nvcc}}
