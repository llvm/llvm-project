/* Checks that BOLT correctly processes a user-provided function list file,
 * reorder functions according to this list, update hot_start and hot_end
 * symbols and insert a function to perform hot text mapping during program
 * startup.
 */
#include <stdio.h>

int foo(int x) {
  return x + 1;
}

int fib(int x) {
  if (x < 2)
    return x;
  return fib(x - 1) + fib(x - 2);
}

int bar(int x) {
  return x - 1;
}

int main(int argc, char **argv) {
  printf("fib(%d) = %d\n", argc, fib(argc));
  return 0;
}

/*
REQUIRES: system-linux,bolt-runtime

RUN: %clang %cflags %s -o %t.exe -Wl,-q

RUN: llvm-bolt %t.exe -relocs=1 -lite -reorder-functions=user \
RUN:   -hugify -function-order=%p/Inputs/user_func_order.txt -o %t
RUN: llvm-nm --numeric-sort --print-armap %t | \
RUN:   FileCheck %s -check-prefix=CHECK-NM
RUN: %t 1 2 3 | FileCheck %s -check-prefix=CHECK-OUTPUT

CHECK-NM:      W  __hot_start
CHECK-NM:      T main
CHECK-NM-NEXT: T fib
CHECK-NM-NEXT: W __hot_end

CHECK-OUTPUT: fib(4) = 3
*/
