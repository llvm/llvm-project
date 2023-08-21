// XFAIL:*
//// Currently debug info for 'local' behaves, but 'plocal' dereferences to
//// the incorrect value 0xFF after the call to esc.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %dexter --fail-lt 1.0 -w --debugger lldb \
// RUN:     --builder clang-c --cflags "-O2 -glldb" -- %s
//
//// Check that a pointer to a variable living on the stack dereferences to the
//// variable value.

int glob;
__attribute__((__noinline__))
void esc(int* p) {
  glob = *p;
  *p = 0xFF;
}

int main() {
  int local = 0xA;
  int *plocal = &local;
  esc(plocal);      // DexLabel('s1')
  local = 0xB;      //// DSE
  return 0;         // DexLabel('s2')
}


// DexExpectWatchValue('local', 0xA, on_line=ref('s1'))
// DexExpectWatchValue('local', 0xB, on_line=ref('s2'))
// DexExpectWatchValue('*plocal', 0xA, on_line=ref('s1'))
// DexExpectWatchValue('*plocal', 0xB, on_line=ref('s2'))
//// Ideally we should be able to observe the dead store to local (0xB) through
//// plocal here.
// DexExpectWatchValue('(local == *plocal)', 'true', from_line=ref('s1'), to_line=ref('s2'))
