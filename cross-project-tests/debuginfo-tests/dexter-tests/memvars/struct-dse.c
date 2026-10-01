// XFAIL:*
//// Currently, LowerDbgDeclare doesn't lower dbg.declares pointing at allocas
//// for structs.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O2 -glldb %s -o %t
// RUN: %dexter -w %dexter_lldb_args --binary %t -- %s | FileCheck %s
//
//// Check debug-info for the escaped struct variable num is reasonable.

#include <stdio.h>
struct Nums {
  int a, b, c, d, e, f, g, h, i, j;
};
struct Nums glob;
__attribute__((__noinline__))
void esc(struct Nums* nums) {
  glob = *nums;
}

__attribute__((__noinline__))
int main() {
  struct Nums nums = { .c=1 };       //// Dead store.
  printf("s1 nums.c: %d\n", nums.c); // !dex_label s1

  nums.c = 2;                        //// Killing store.
  printf("s2 nums.c: %d\n", nums.c); // !dex_label s2

  esc(&nums);                        //// Force nums to live on the stack.
  return 0;                          // !dex_label s3
}

// CHECK-DAG: seen_values: 2
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {lines: !label s1}:
  !value nums:
    c: 1
!where {lines: !range [!label s2, !label s3]}:
  !value nums:
    c: 2
...
*/
