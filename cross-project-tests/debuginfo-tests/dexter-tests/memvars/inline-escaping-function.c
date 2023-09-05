// XFAIL: *
// Incorrect location for variable "param", see PR48719.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %dexter --fail-lt 1.0 -w --debugger lldb \
// RUN:     --builder 'clang-c'  --cflags "-O3 -glldb" -- %s

// 1. param is escaped by inlineme(&param) so it is not promoted by
//    SROA/mem2reg.
// 2. InstCombine's LowerDbgDeclare converts the dbg.declare to a set of
//    dbg.values.
// 3. inlineme(&param) is inlined.
// 4. SROA/mem2reg fully promotes param. It does not insert a dbg.value after the
//    PHI it inserts which merges the values out of entry and if.then in the
//    sucessor block. This behaviour is inconsistent. If the dbg.declare was
//    still around (i.e.  if param was promoted in the first round of mem2reg
//    BEFORE LowerDbgDeclare) we would see a dbg.value insered for the PHI.
// 5. JumpThreading removes the if.then block, changing entry to
//    unconditionally branch to if.end.
// 6. SimplifyCFG stitches entry and if.end together.

// The debug info is not updated to account for the merged value prior to or
// during JumpThreading/SimplifyCFG so we end up seeing param=5 for the entire
// function, when we'd expect to see param=10 when stepping onto fluff().

__attribute__((always_inline))
int inlineme(int* p) { return *p * 2; }

__attribute__((optnone))
void fluff() {}

__attribute__((noinline))
int fun(int param) {
  if (param)
    param = inlineme(&param);
  fluff();           // DexLabel('s0')
  return param;
}

int main() {
  return fun(5);
}

// DexExpectWatchValue('param', 10, on_line=ref('s0'))
