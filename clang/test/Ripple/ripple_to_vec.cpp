// REQUIRES: target=aarch64{{.*}} || target=x86_64{{.*}}
// RUN: %clang %s -O2 -fenable-ripple -S -emit-llvm -o - | FileCheck %s

#include <ripple.h>
#include <ripple/zip.h>

typedef int32_t __attribute__((vector_size(128))) Vector_t;
typedef int32_t __attribute__((vector_size(256))) VectorPair_t;

extern "C" {

Vector_t norm(VectorPair_t pair) {
// CHECK: define dso_local{{.*}}@norm
  ripple_block_t BS = ripple_set_block_shape(0, 32, 2);
  int32_t x = vec_to_ripple_2d<64, int32_t>(BS, pair);
  // This "zip" shuffle turned a sequence of pairs into a pair of sequences
  int32_t even_odds = ripple_shuffle(x, rzip::shuffle_unzip<2, 0, 0>);
  int32_t evens = ripple_slice(even_odds, -1, 0); // 32x1 shape
  int32_t odds = ripple_slice(even_odds, -1, 1); // 32x1 shape
  return ripple_to_vec<32, int32_t>(BS, evens * odds);
}


// Same, with default pe_id
Vector_t norm2(VectorPair_t pair) {
// CHECK: define dso_local{{.*}}@norm2
  ripple_block_t BS = ripple_set_block_shape(0, 32, 2);
  int32_t x = vec_to_ripple_2d<64, int32_t>(BS, pair);
  // This "zip" shuffle turned a sequence of pairs into a pair of sequences
  int32_t even_odds = ripple_shuffle(x, rzip::shuffle_unzip<2, 0, 0>);
  int32_t evens = ripple_slice(even_odds, -1, 0); // 32x1 shape
  int32_t odds = ripple_slice(even_odds, -1, 1); // 32x1 shape
  return ripple_to_vec<32, int32_t>(BS, evens * odds);
}

}