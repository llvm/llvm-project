// XFAIL: target={{.*(iu|riscv).*}}
// RUN: %clang++ -g -S -fenable-ripple -O2 -emit-llvm %s -o - 2>&1 | FileCheck %s --implicit-check-not="warning:" --implicit-check-not="error:"

#include <ripple.h>

// CHECK: ripple_shuffle_cpp_lambda_without_capture
void ripple_shuffle_cpp_lambda_without_capture(size_t ArraySize, float *Input,
                                               float *Output) {
  ripple_block_t BS = ripple_set_block_shape(0, 1);
  unsigned BlockIdx = ripple_id(BS, 0);
  unsigned BlockSize = ripple_get_block_size(BS, 0);
  for (size_t Idx = 0; Idx < ArraySize; Idx += BlockSize) {
    float Tmp = Input[Idx + BlockIdx];
    float Shuffled =
        ripple_shuffle(Tmp, [](size_t Idx, size_t BlockSz) -> size_t {
          return Idx + 1 % BlockSz;
        });
    Output[Idx + BlockIdx] = Shuffled;
  }
}

// The compiler does not seem to be able to locate the <functional> header (or
// any c++ headers for Hexagon)!
#if 0
void ripple_shuffle_cpp_lambda_with_capture(size_t ArraySize, float *Input,
                                            float *Output) {
  ripple_block_t BS = ripple_set_block_shape(0, 1);
  unsigned BlockIdx = ripple_id(BS, 0);
  unsigned BlockSize = ripple_get_block_size(BS, 0);
  for (size_t Idx = 0; Idx < ArraySize; Idx += BlockSize) {
    float Tmp = Input[Idx + BlockIdx];
    float Shuffled =
        ripple_shuffle(Tmp, [&BlockSize](size_t Idx, size_t _) -> size_t {
          return Idx + 1 % BlockSize;
        });
    Output[Idx + BlockIdx] = Shuffled;
  }
}
#endif
