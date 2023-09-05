// REQUIRES: lldb
// UNSUPPORTED: system-windows
//
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder clang-c --debugger 'lldb' --cflags "-O -glldb" -- %s

void __attribute__((noinline, optnone)) bar(int *test) {}
int main() {
  int test;
  test = 23;
  bar(&test); // DexLabel('before_bar')
  return test; // DexLabel('after_bar')
}

// DexExpectWatchValue('test', '23', on_line=ref('before_bar'))
// DexExpectWatchValue('test', '23', on_line=ref('after_bar'))

