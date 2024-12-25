// REQUIRES: nvptx-registered-target
// REQUIRES: x86-registered-target

// RUN: %clang_cc1 "-triple" "x86_64-unknown-linux-gnu" "-aux-triple" "nvptx64-nvidia-cuda" \
// RUN:    "-target-cpu" "x86-64"  "-fsyntax-only" %s

#include "Inputs/cuda.h"

typedef __attribute__((ext_vector_type(16))) float float32x16_t;
__device__ void test(float32x16_t& vodd) {
  constexpr int pose = 16;
  __asm__ __volatile__("vadd %0, %1, %2":"=&v"(vodd):"r"(pose),"v"(vodd));
}
