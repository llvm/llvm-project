// RUN: %clangxx -DOP=n++ -fsanitize=unsigned-integer-overflow %s -o %t1 && %run %t1 2>&1 | FileCheck --check-prefix=CHECK-INC %s
// RUN: %clangxx -DOP=++n -fsanitize=unsigned-integer-overflow %s -o %t2 && %run %t2 2>&1 | FileCheck --check-prefix=CHECK-INC %s
// RUN: %clangxx -DOP=m-- -fsanitize=unsigned-integer-overflow %s -o %t3 && %run %t3 2>&1 | FileCheck --check-prefix=CHECK-DEC %s
// RUN: %clangxx -DOP=--m -fsanitize=unsigned-integer-overflow %s -o %t4 && %run %t4 2>&1 | FileCheck --check-prefix=CHECK-DEC %s

#include <stdint.h>

int main() {
  unsigned n = 0xfffffffd;
  n++;
  n++;
  unsigned m = 0;
  // CHECK-INC: uincdec-overflow.cpp:15:3: runtime error: unsigned integer overflow: 4294967295 + 1 cannot be represented in type 'unsigned int'
  // CHECK-DEC: uincdec-overflow.cpp:15:3: runtime error: unsigned integer overflow: 0 - 1 cannot be represented in type 'unsigned int'
  OP;
}
