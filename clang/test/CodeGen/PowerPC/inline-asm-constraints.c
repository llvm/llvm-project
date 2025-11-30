// RUN: %clang_cc1 -emit-llvm -triple powerpc64-ibm-aix-xcoff \
// RUN:   %s -o - | FileCheck %s

#include <stdint.h>

// Test Atomic Memory Operation Support:
// This test case takes an address and performs an atomic load at that address.
// The purpose is to test the Q machine constraint and P machine constraint
// argument modifier together.
// These constraints on the pointer `ptr` read as: constrain (uint32_t*)ptr to
// read and writeable X-Form Addressed Memory operands.
static __attribute__((noinline))
uint32_t atomic_load(uint32_t *ptr, uint32_t val)
{
// CHECK-LABEL: define{{.*}} i32 @atomic_load(ptr noundef %ptr, i32 noundef zeroext %val)
// CHECK:  %3 = call { i128, i32 } asm sideeffect "mr ${1:L},$3\0A\09 lwat $1,${0:P},$4\0A\09 mr $2,$1\0A", "=*Q,=&r,=r,r,n,0"(ptr elementtype(i32) %arrayidx, i32 %2, i32 0, i32 %1)
  unsigned __int128 tmp;
  uint32_t ret;
  __asm__ volatile ("mr %L1,%3\n"
                    "\t lwat %1,%P0,%4\n"
                    "\t mr %2,%1\n"
                    : "+Q" (ptr[0]), "=&r" (tmp), "=r" (ret)
                    : "r" (val), "n" (0x00));
  return ret;
}

int main(int argc, char **argv) {
   return atomic_load((uint32_t*)argv[1], (uint32_t)*(argv[2]));
}