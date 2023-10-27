// Location for variable "parama" optimized out.
// Previously it would carry incorrect location
// information in debug-info, see PR48719.
// Now, the location is simply not emitted.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O3 -glldb %s -o %t
// RUN: %dexter --fail-lt 0.1 -w --debugger lldb --binary %t -- %s
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
  fluff();            // DexLabel('s0')
  return paramb;
}

int main() {
  return fun(5, 20);
}

// DexExpectWatchValue('parama', 20, on_line=ref('s0'))
//
// NOTE: the dexter command uses --fail-lt 0.1 (instead of the standard 1.0)
// because seeing 'optimized out' would still be a win; it's the best we can do
// without using conditional DWARF operators in the location expression. Seeing
// 'optimized out' should result in a score higher than 0.1.
