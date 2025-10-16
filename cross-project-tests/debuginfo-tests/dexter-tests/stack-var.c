// REQUIRES: lldb
// UNSUPPORTED: system-windows
//
// RUN: %clang -std=gnu11 -O -glldb %s -o %t
// RUN: %dexter --fail-lt 1.0 -w --binary %t %dexter_lldb_args -- %s

void __attribute__((noinline, optnone)) bar(int *test) {}
int main() {
  int test;
  test = 23;
  bar(&test); // DexLabel('before_bar')
  return test; // DexLabel('after_bar')
}

// DexExpectWatchValue('test', '23', on_line=ref('before_bar'))
// DexExpectWatchValue('test', '23', on_line=ref('after_bar'))

