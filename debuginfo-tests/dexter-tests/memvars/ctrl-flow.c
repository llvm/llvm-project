// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %dexter --fail-lt 1.0 -w --debugger lldb \
// RUN:     --builder clang-c  --cflags "-O2 -glldb" -- %s

//// Check that we give good locations to a variable ('local') which is escaped
//// down some control paths and not others. This example is handled well currently.

int g;
__attribute__((__noinline__))
void leak(int *ptr) {
  g = *ptr;
  *ptr = 2;
}

__attribute__((__noinline__))
int fun(int cond) {
  int local = 0;   // DexLabel('s1')
  if (cond)
    leak(&local);
  else
    local = 1;
  return local;    // DexLabel('s2')
}

int main() {
  int a = fun(1);
  int b = fun(0);
  return a + b;
}

////                           fun(1)  fun(0)
// DexExpectWatchValue('local',   '0',    '0', on_line=ref('s1'))
// DexExpectWatchValue('local',   '2',    '1', on_line=ref('s2'))
