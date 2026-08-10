// RUN: %clangxx -O0 -g %s -o %t && %run %t 2>&1 | FileCheck %s

#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
  unsigned char key[] = {0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99};
  uint32_t hashes[3];
  mi_vector_hash(key, __arraycount(key), 0, hashes);
  for (size_t i = 0; i < __arraycount(hashes); i++)
    printf("hashes[%zu]='%" PRIx32 "'\n", i, hashes[i]);

  // CHECK: hashes[0]='{{.*}}'
  // CHECK: hashes[1]='{{.*}}'
  // CHECK: hashes[2]='{{.*}}'

  return 0;
}
