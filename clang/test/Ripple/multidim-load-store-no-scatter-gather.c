// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -g -S -fenable-ripple -O1 -emit-llvm %s -o - 2>&1 | FileCheck %s
// RUN: %clang -g -S -fenable-ripple -O2 -emit-llvm %s -o - 2>&1 | FileCheck %s
// RUN: %clang -g -S -fenable-ripple -O3 -emit-llvm %s -o - 2>&1 | FileCheck %s
// RUN: %clang -g -S -fenable-ripple -Os -emit-llvm %s -o - 2>&1 | FileCheck %s
// RUN: %clang -g -S -fenable-ripple -Oz -emit-llvm %s -o - 2>&1 | FileCheck %s

#include <stddef.h>
#include <ripple.h>

// CHECK-NOT: masked.gather
// CHECK-NOT: masked.scatter

void test1(size_t num, float (*In)[32], float (*Out)[32]) {
  ripple_block_t BS = ripple_set_block_shape(0, 8, 4);
  size_t BlockIdx = ripple_id(BS, 0);
  size_t BlockIdx1 = ripple_id(BS, 1);
  size_t BlockSize = ripple_get_block_size(BS, 0);
  for (unsigned i = 0; i < num; ++i)
    Out[i][BlockIdx + BlockIdx1 * BlockSize] = In[i][BlockIdx + BlockIdx1 * BlockSize];
}

void test2(size_t num, float (*In)[32], float (*Out)[32]) {
  ripple_block_t BS = ripple_set_block_shape(0, 8, 4);
  for (unsigned i = 0; i < num; ++i)
    ripple_parallel_full(BS, 0, 1);
    for (size_t j = 0; j < 32; j++)
      Out[i][j] = In[i][j];
}

void test3(size_t num, float (*In)[32], float (*Out)[32]) {
  ripple_block_t BS = ripple_set_block_shape(0, 8, 4);
  for (unsigned i = 0; i < num; ++i)
    ripple_parallel_full(BS, 0, 1);
    for (int j = 0; j < 32; j++)
      Out[i][j] = In[i][j];
}

void test4(size_t num, float (*In)[32], float (*Out)[32]) {
  ripple_block_t BS = ripple_set_block_shape(0, 8, 4);
  for (unsigned i = 0; i < num; ++i)
    ripple_parallel_full(BS, 0, 1);
    for (int32_t j = 0; j < 32; j++)
      Out[i][j] = In[i][j];
}

void test5(size_t num, float (*In)[32], float (*Out)[32]) {
  ripple_block_t BS = ripple_set_block_shape(0, 8, 2, 2);
  for (unsigned i = 0; i < num; ++i)
    ripple_parallel_full(BS, 0, 1, 2);
    for (int j = 0; j < 32; j++)
      Out[i][j] = In[i][j];
}

void test6(size_t num, float (*In)[32], float (*Out)[32]) {
  ripple_block_t BS = ripple_set_block_shape(0, 8, 2, 2);
  for (unsigned i = 0; i < num; ++i)
    ripple_parallel_full(BS, 0, 1, 2);
    for (size_t j = 0; j < 32; j++)
      Out[i][j] = In[i][j];
}

/*

// We need to discuss if we want to handle unsigned w/ the non-undefined behavior semantics!
void test7(size_t num, float (*In)[32], float (*Out)[32]) {
  ripple_block_t BS = ripple_set_block_shape(0, 8, 2, 2);
  for (unsigned i = 0; i < num; ++i)
    ripple_parallel_full(BS, 0, 1, 2);
    for (uint32_t j = 0; j < 32; j++)
      Out[i][j] = In[i][j];
}

*/
