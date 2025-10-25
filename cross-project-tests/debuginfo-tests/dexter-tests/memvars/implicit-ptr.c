// XFAIL:*
//// We don't yet support DW_OP_implicit_pointer in llvm.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O3 -glldb %s -o %t
// RUN: %dexter --fail-lt 1.0 -w %dexter_lldb_args --binary %t -- %s

//// Check that 'param' in 'fun' can be read throughout, and that 'pa' and 'pb'
//// can be dereferenced in the debugger even if we can't provide the pointer
//// value itself.

int globa;
int globb;

//// A no-inline, read-only function with internal linkage is a good candidate
//// for arg promotion.
__attribute__((__noinline__))
static void use_promote(const int* pa) {
  //// Promoted args would be a good candidate for an DW_OP_implicit_pointer.
  globa = *pa; // DexLabel('s2')
}

__attribute__((__always_inline__))
static void use_inline(const int* pb) {
  //// Inlined pointer to callee local would be a good candidate for an
  //// DW_OP_implicit_pointer.
  globb = *pb; // DexLabel('s3')
}

__attribute__((__noinline__))
int fun(int param) {
  volatile int step = 0;   // DexLabel('s1')
  use_promote(&param);
  use_inline(&param);
  return step;             // DexLabel('s4')
}

int main() {
  return fun(5);
}

// DexExpectWatchValue('param', 5, from_line=ref('s1'), to_line=ref('s4'))
// DexExpectWatchValue('*pa', 5, on_line=ref('s2'))
// DexExpectWatchValue('*pb', 5, on_line=ref('s3'))
