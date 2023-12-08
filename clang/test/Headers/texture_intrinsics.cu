// REQUIRES: x86-registered-target
// REQUIRES: nvptx-registered-target
//
// RUN: %clang -std=c++11 -fsyntax-only -target x86_64-linux -nocudainc -nocudalib --cuda-gpu-arch=sm_86 --cuda-device-only -S %s
// RUN: %clang -std=c++11 -fsyntax-only -target x86_64-linux -nocudainc -nocudalib --cuda-gpu-arch=sm_86 --cuda-host-only -S %s

// Define bare minimum required for parsing the header file.
#include "Inputs/include/cuda.h"

// The header file is expected to compile w/o errors.  This ensures that texture
// ID hash has no collisions for known texture operations, otherwise the
// compilation would fail with an attempt to redefine a type.
#include <__clang_cuda_texture_intrinsics.h>
