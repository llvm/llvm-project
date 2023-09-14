// Purpose:
// Ensure that debug information for a local variable does not hide
// a global definition that has the same name.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// XFAIL: system-darwin
// RUN: %clang -std=gnu++11 -O0 -g %s -o %t
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --binary  %t --debugger 'lldb' -v -- %s

const int d = 100;

extern int foo();

int main() {
  const int d = 4;
  const float e = 4; // DexLabel("main")
  const char *f = "Woopy";
  return d + foo();
}

int foo() {
  return d; // DexLabel("foo")
}

// DexExpectWatchValue('d', '4', on_line=ref('main'))
// DexExpectWatchValue('d', '100', on_line=ref('foo'))

