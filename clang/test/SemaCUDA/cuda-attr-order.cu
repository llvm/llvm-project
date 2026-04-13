// Verify that we can parse a simple CUDA file with different attributes order.
// RUN: %clang_cc1 "-triple" "nvptx-nvidia-cuda"  -fsyntax-only -verify %s
// expected-no-diagnostics
#include "Inputs/cuda.h"

struct alignas(16) float4 {
    float x, y, z, w;
};

__attribute__((device)) float func() {
    __shared__ alignas(alignof(float4)) float As[4][4];  // Both combinations
    alignas(alignof(float4)) __shared__  float Bs[4][4]; // must be legal

    return As[0][0] + Bs[0][0];
}
