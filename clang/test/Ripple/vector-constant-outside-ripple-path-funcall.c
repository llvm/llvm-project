// REQUIRES: target-x86_64 || target-aarch64 || target=hexagon{{.*}}
// RUN: %clang -fenable-ripple -g -S -O2 -emit-llvm %s

#include <ripple.h>
#include <stddef.h>

typedef uint32_t u32t32 __attribute__((__vector_size__(128)))
__attribute__((aligned(128)));

extern u32t32 externF(u32t32);

void f(uint32_t *Input, uint32_t *sr){
  ripple_block_t BS = ripple_set_block_shape(0, 32);
  int v0 = ripple_id(BS, 0);
  uint32_t Vacc_0_32[32];
  for (size_t i = 0; i < 32; ++i)
    Vacc_0_32[i] = 0;
  u32t32 Vacc_2 = externF(*((u32t32*)Vacc_0_32));
  *((u32t32*)sr) = Vacc_2;
  sr[v0] += Input[v0];
}
