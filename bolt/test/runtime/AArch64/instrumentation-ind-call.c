#include <stdio.h>

typedef int (*func_ptr)(int, int);

int add(int a, int b) { return a + b; }

int main() {
  func_ptr fun;
  fun = add;
  int sum = fun(10, 20); // indirect call to 'add'
  printf("The sum is: %d\n", sum);
  return 0;
}
/*
REQUIRES: system-linux,bolt-runtime

RUN: %clang %cflags %s -o %t.exe -Wl,-q -no-pie -fpie

RUN: llvm-bolt %t.exe --instrument --instrumentation-file=%t.fdata \
RUN:   -o %t.instrumented

# Instrumented program needs to finish returning zero
RUN: %t.instrumented | FileCheck %s -check-prefix=CHECK-OUTPUT

# Test that the instrumented data makes sense
RUN:  llvm-bolt %t.exe -o %t.bolted --data %t.fdata \
RUN:    --reorder-blocks=ext-tsp --reorder-functions=hfsort+ \
RUN:    --print-only=main --print-finalized | FileCheck %s

RUN: %t.bolted | FileCheck %s -check-prefix=CHECK-OUTPUT

CHECK-OUTPUT: The sum is: 30

# Check that our indirect call has 1 hit recorded in the fdata file and that
# this was processed correctly by BOLT
CHECK:         blr     x8 # CallProfile: 1 (0 misses) :
CHECK-NEXT:    { add: 1 (0 misses) }
*/
