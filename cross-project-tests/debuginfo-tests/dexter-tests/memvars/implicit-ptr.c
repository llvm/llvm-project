// XFAIL:*
//// We don't yet support DW_OP_implicit_pointer in llvm.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O3 -glldb %s -o %t
// RUN: %dexter -w --use-script %dexter_lldb_args --binary %t -- %s \
// RUN:   | FileCheck %s

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
  globa = *pa; // !dex_label s2
}

__attribute__((__always_inline__))
static void use_inline(const int* pb) {
  //// Inlined pointer to callee local would be a good candidate for an
  //// DW_OP_implicit_pointer.
  globb = *pb; // !dex_label s3
}

__attribute__((__noinline__))
int fun(int param) {
  volatile int step = 0;   // !dex_label s1
  use_promote(&param);
  use_inline(&param);
  return step;             // !dex_label s4
}

int main() {
  return fun(5);
}

// CHECK-DAG: seen_values: 3
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {lines: !range [!label s1, !label s4]}:
  !value param: 5
!where {lines: !label s2}:
  !value pa:
    "*": 5
!where {lines: !label s3}:
  !value pa:
    "*": 5
...
*/
