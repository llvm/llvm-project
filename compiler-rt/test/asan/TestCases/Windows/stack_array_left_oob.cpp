// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: not %run %t 2>&1 | FileCheck %s

#include <stdio.h>

int main() {
  int subscript = -1;
  char buffer[42];
  buffer[subscript] = 42;
// CHECK: AddressSanitizer: stack-buffer-underflow on address [[ADDR:0x[0-9a-f]+]]
// CHECK: WRITE of size 1 at [[ADDR]] thread T0
// CHECK-NEXT: {{#0 .* main .*stack_array_left_oob.cpp}}:[[@LINE-3]]
// CHECK: Address [[ADDR]] is located in stack of thread T0 at offset [[OFFSET:.*]] in frame
// CHECK-NEXT: {{#0 .* main .*stack_array_left_oob.cpp}}
// CHECK: 'buffer'{{.*}} <== Memory access at offset [[OFFSET]] underflows this variable
}
