// XFAIL:*
//// Currently debug info for 'local' behaves, but 'plocal' dereferences to
//// the incorrect value 0xFF after the call to esc.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// RUN: %clang -std=gnu11 -O2 -glldb %s -o %t
// RUN: %dexter -w --use-script %dexter_lldb_args --binary %t -- %s \
// RUN:   | FileCheck %s
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
  esc(plocal);      // !dex_label s1
  local = 0xB;      //// DSE
  return 0;         // !dex_label s2
}

// CHECK-DAG: seen_values: 5
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {lines: !label s1}:
  !value local: 0xA
  !value plocal:
    "*": 0xA
!where {lines: !label s2}:
  !value local: 0xB
  !value plocal:
    "*": 0xB
# Ideally we should be able to observe the dead store to local (0xB) through
# plocal here.
!where {lines: !range [!label s1, !label s2]}:
  !value "(local == *plocal)": true
...
*/
