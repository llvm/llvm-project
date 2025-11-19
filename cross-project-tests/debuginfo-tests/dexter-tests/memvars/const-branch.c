// XFAIL: !system-darwin || !target-aarch64
//// Suboptimal coverage, see inlined comments.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O3 -glldb %s -o %t
// RUN: %dexter --fail-lt 1.0 -w %dexter_lldb_args --binary %t -- %s

//// Adapted from https://bugs.llvm.org/show_bug.cgi?id=34136#c4

int g;

__attribute__((__noinline__))
void esc(int* p) {
  g = *p;
  *p = 5;
}

__attribute__((__noinline__))
void thing(int x) {
  g = x;
}

__attribute__((__noinline__))
int fun(int param) {
  esc(&param);      //// alloca is live until here        DexLabel('s1')
  if (param == 0) { //// end of alloca live range
    //// param is now a constant, but without lowering to dbg.value we can't
    //// capture that and would still point to the stack slot that may even have
    //// been reused by now.
    ////
    //// Right now we get suboptimal coverage for x86: the param load below is
    //// CSE'd with the if condition.
    //// Instcombine runs LowerDbgDeclare and inserts a dbg.value after the load.
    //// SelectionDAG combines the load and cmp. We go from this IR:
    ////   %0 = load i32, i32* %param.addr, align 4, !dbg !42, !tbaa !20
    ////   call void @llvm.dbg.value(metadata i32 %0, ...
    ////   %cmp = icmp eq i32 %0, 0, !dbg !44
    //// to this MIR:
    ////   DBG_VALUE $noreg, $noreg, !"param"...
    ////   CMP32mi8 %param.addr, 1, $noreg, 0, $noreg, 0, implicit-def $eflags, debug-location !44
    thing(param);
  }
  return 0; //                                            DexLabel('s2')
}

int main() {
  return fun(5);
}

// DexExpectWatchValue('param', '5',  from_line=ref('s1'), to_line=ref('s2'))

