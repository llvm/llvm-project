// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang++ -O0 -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"
// RUN: %clang++ -O1 -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"
// RUN: %clang++ -O2 -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"
// RUN: %clang++ -O3 -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"
// RUN: %clang++ -Og -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"
// RUN: %clang++ -Os -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"
// RUN: %clang++ -Oz -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"

#include <ripple.h>

// Checking that the return attributes ZeroExtend and SignExtend are removed for vector types during specialization.

// CHECK-NOT: zeroext{{.*}}ripple.specialization
// CHECK-NOT: signext{{.*}}ripple.specialization

namespace mymax {
  template <typename T>
  const T max(const T a, const T b) {
      return (a < b) ? b : a;
  }
}

void check(const uint8_t  *input, uint8_t  *output) {
  auto BS = ripple_set_block_shape(0, 32);
  size_t v0 = ripple_id(BS, 0);
  uint8_t max = input[v0];
  max = mymax::max(max, input[v0 + 32]);
  output[v0] = max;
  int8_t maxi = input[v0];
  maxi = mymax::max(maxi, (int8_t)input[v0 + 32]);
  output[v0 + 32] = maxi;
}
