// XFAIL:*
//// See PR47946.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %dexter --fail-lt 1.0 -w --debugger lldb \
// RUN:     --builder clang-c  --cflags "-O2 -glldb" -- %s
//
//// Check that once-escaped variable 'param' can still be read after we
//// perform inlining + mem2reg, and that we see the DSE'd value 255.


int g;
__attribute__((__always_inline__))
static void use(int* p) {
  g = *p;
  *p = 255;
  volatile int step = 0;  // DexLabel('use1')
}

__attribute__((__noinline__))
void fun(int param) {
  //// Make sure first step is in 'fun'.
  volatile int step = 0;  // DexLabel('fun1')
  use(&param);
  return;                 // DexLabel('fun2')
}

int main() {
  fun(5);
}

/*
# Expect param == 5 before stepping through inlined 'use'.
DexExpectWatchValue('param', '5', on_line='fun1')

# Expect param == 255 after assignment in inlined frame 'use'.
DexExpectProgramState({
  'frames': [
    { 'function': 'use',
      'location': { 'lineno': 'use1' },
    },
    { 'function': 'fun',
      'location': { 'lineno': 20 },
      'watches':  { 'param': '255' }
    },
  ]
})

# Expect param == 255 after inlined call to 'use'.
DexExpectWatchValue('param', '255', on_line='fun2')
*/
