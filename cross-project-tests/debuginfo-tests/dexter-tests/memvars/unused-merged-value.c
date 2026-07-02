// Location for variable "parama" optimized out.
// Previously it would carry incorrect location
// information in debug-info, see PR48719.
// Now, the location is simply not emitted.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O3 -glldb %s -o %t
// RUN: %dexter -w --use-script %dexter_lldb_args --binary %t -- %s \
// RUN:   | FileCheck %s
// See NOTE at end for more info about the RUN command.

// 1. SROA/mem2reg fully promotes parama.
// 2. parama's value in the final block is the merge of values for it coming
//    out of entry and if.then. If the variable were used later in the function
//    mem2reg would insert a PHI here and add a dbg.value to track the merged
//    value in debug info. Because it is not used there is no PHI (the merged
//    value is implicit) and subsequently no dbg.value.
// 3. SimplifyCFG later folds the blocks together (if.then does nothing besides
//    provide debug info so it is removed and if.end is folded into the entry
//    block).

// The debug info is not updated to account for the implicit merged value prior
// to (e.g. during mem2reg) or during SimplifyCFG so we end up seeing parama=5
// for the entire function, which is incorrect.

__attribute__((optnone))
void fluff() {}

__attribute__((noinline))
int fun(int parama, int paramb) {
  if (parama)
    parama = paramb;
  fluff(); // !dex_label s0
  return paramb;
}

int main() {
  return fun(5, 20);
}

// NOTE: we check for optimized_out_steps instead of correct_steps, because
// parama being 'optimized out' instead of missing is the best we can do without
// using conditional DWARF operators in the location expression. Therefore, this
// test will still pass if we see "optimized out" instead of "missing".
// If we ever manage to recover this variable information, then we can update
// this test to expect correctness.

// CHECK: optimized_out_steps: 1
// CHECK: missing_var_steps: 0

/*
---
!where {lines: !label s0}:
  !value parama: 20
...
*/
