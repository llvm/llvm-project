// REQUIRES: librt_has_emupac
// RUN: %clang_builtins %s %librt -o %t
// RUN: %run %t 1
// RUN: %run %t 2
// RUN: %expect_crash %run %t 3
// RUN: %expect_crash %run %t 4

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

uint64_t __emupac_pacda(uint64_t ptr, uint64_t disc);
uint64_t __emupac_autda(uint64_t ptr, uint64_t disc);

int main(int argc, char **argv) {
  char stack_object1;
  uint64_t ptr1 = (uint64_t)&stack_object1;

  char stack_object2;
  uint64_t ptr2 = (uint64_t)&stack_object2;

  switch (atoi(argv[1])) {
  case 1: {
    // Normal case: test that a pointer authenticated with the same
    // discriminator is equal to the original pointer.
    uint64_t signed_ptr = __emupac_pacda(ptr1, ptr2);
    uint64_t authed_ptr = __emupac_autda(signed_ptr, ptr2);
    if (authed_ptr != ptr1) {
      printf("0x%lx != 0x%lx\n", authed_ptr, ptr1);
      return 1;
    }
    break;
  }
  case 2: {
    // Test that negative addresses (addresses controlled by TTBR1,
    // conventionally kernel addresses) can be signed and authenticated.
    uint64_t unsigned_ptr = -1ULL;
    uint64_t signed_ptr = __emupac_pacda(unsigned_ptr, ptr2);
    uint64_t authed_ptr = __emupac_autda(signed_ptr, ptr2);
    if (authed_ptr != unsigned_ptr) {
      printf("0x%lx != 0x%lx\n", authed_ptr, unsigned_ptr);
      return 1;
    }
    break;
  }
  case 3: {
    // Test that a corrupted signature crashes the program.
    uint64_t signed_ptr = __emupac_pacda(ptr1, ptr2);
    __emupac_autda(signed_ptr + (1ULL << 48), ptr2);
    break;
  }
  case 4: {
    // Test that signing a pointer with signature bits already set produces a pointer
    // that would fail auth.
    uint64_t signed_ptr = __emupac_pacda(ptr1 + (1ULL << 48), ptr2);
    __emupac_autda(signed_ptr, ptr2);
    break;
  }
  }

  return 0;
}
