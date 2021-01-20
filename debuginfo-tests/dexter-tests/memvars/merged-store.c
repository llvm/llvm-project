// XFAIL: *
// Incorrect location for variable "parama", see PR48719.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %dexter --fail-lt 1.0 -w --debugger lldb \
// RUN:     --builder 'clang-c'  --cflags "-O3 -glldb" -- %s

// 1. parama is escaped by esc(&parama) so it is not promoted by
//    SROA/mem2reg.
// 2. InstCombine's LowerDbgDeclare converts the dbg.declare to a set of
//    dbg.values (tracking the stored SSA values).
// 3. InstCombine replaces the two stores to parama's alloca (the initial
//    parameter register store in entry and the assignment in if.then) with a
//    PHI+store in the common sucessor.
// 4. SimplifyCFG folds the blocks together and converts the PHI to a
//    select.

// The debug info is not updated to account for the merged value in the
// sucessor prior to SimplifyCFG when it exists as a PHI, or during when it
// becomes a select. As a result we see parama=5 for the entire function, when
// we'd expect to see param=20 when stepping onto fluff().

__attribute__((optnone))
void esc(int* p) {}

__attribute__((optnone))
void fluff() {}

__attribute__((noinline))
int fun(int parama, int paramb) {
  if (parama)
    parama = paramb;
  fluff();           // DexLabel('s0')
  esc(&parama);
  return 0;
}

int main() {
  return fun(5, 20);
}

// DexExpectWatchValue('parama', 20, on_line='s0')
