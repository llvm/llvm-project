// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang++ -O0 -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"
// RUN: %clang++ -O1 -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"
// RUN: %clang++ -O2 -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"
// RUN: %clang++ -O3 -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"
// RUN: %clang++ -Og -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"
// RUN: %clang++ -Os -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"
// RUN: %clang++ -Oz -Wall -Wextra -Wripple -S -emit-llvm -fenable-ripple %s -o - | FileCheck %s --implicit-check-not="warning:"

#include <ripple.h>

// Checking that this template specialization masking succeeds (it creates a branch and updates the DominatorTree)

// mymax::max() is specialized twice during shape propagation: once entering the loop with arguments
// "(scalar, vector) -> vector" and a second time as "(vector, vector) -> vector" once the PHI has been broadcasted to a tensor.

namespace mymax {
  template <typename T>
  const T max(const T a, const T b) {
      return (a < b) ? b : a;
  }
}

void maxpool_ripple_i8(int h_out, int w_out, const uint8_t  *input, uint8_t  *output) {
  auto BS = ripple_set_block_shape(0, 32);
  size_t id = ripple_id(BS, 0);
  uint8_t max = 0;
  for (int ww = h_out; ww < w_out; ww++)
    max = mymax::max(max, input[ww * ripple_get_block_size(BS, 0) + id]);
  output[id] = max;
}
