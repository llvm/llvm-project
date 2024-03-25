// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O2 -glldb %s -o %t
// RUN: %dexter --fail-lt 1.0 -w --debugger lldb --binary %t -- %s
//
//// Check that the once-escaped variable 'param' can still be read after
//// we perform inlining + mem2reg. See D89810 and D85555.

int g;
__attribute__((__always_inline__))
static void use(int* p) {
  g = *p;
}

__attribute__((__noinline__))
void fun(int param) {
  volatile int step1 = 0;  // DexLabel('s1')
  use(&param);
  volatile int step2 = 0;  // DexLabel('s2')
}

int main() {
  fun(5);
}

// DexExpectWatchValue('param', '5', from_line=ref('s1'), to_line=ref('s2'))
