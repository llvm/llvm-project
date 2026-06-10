// RUN: %clang_csan %s -o %t && %run --threads 64 --blocks 64 %t 2>&1 | FileCheck %s

#include "race.h"

volatile int data;

// CHECK: WARNING: ConcurrencySanitizer: data race
// CHECK: {{Write|Read}} of size 4 at 0x{{[0-9a-f]+}}
// CHECK: #0 {{.*write_read_race\.c:[0-9]+:[0-9]+|.*\?\?}}
// CHECK: Address 0x{{[0-9a-f]+}} is global variable 'data'
int main(void) {
  int sink = 0;
  RACE_UNTIL_FOUND(i) {
    data = i;
    sink += data;
  }
  if (sink == -1)
    __builtin_trap();
  return 0;
}
