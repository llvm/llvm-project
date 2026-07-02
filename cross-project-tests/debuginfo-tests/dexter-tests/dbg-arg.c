// REQUIRES: lldb
// UNSUPPORTED: system-windows
//
// This test case checks debug info during register moves for an argument.
// RUN: %clang -std=gnu11 -m64 -mllvm -fast-isel=false -g %s -o %t
// RUN: %dexter -w --use-script \
// RUN:     --binary %t %dexter_lldb_args -- %s | FileCheck %s
//
// Radar 8412415

struct _mtx
{
  long unsigned int ptr;
  int waiters;
  struct {
    int tag;
    int pad;
  } mtxi;
};

int bar(int, int);

int foobar(struct _mtx *mutex) {
  int r = 1;
  int l = 0; // !dex_label l_assign
  int j = 0;
  do {
    if (mutex->waiters) {
      r = 2;
    }
    j = bar(r, l);
    ++l;
  } while (l < j);
  return r + j;
}

int bar(int i, int j) {
  return i + j;
}

int main() {
  struct _mtx m;
  m.waiters = 0;
  return foobar(&m);
}

// CHECK-DAG: total_watched_steps: 1
// CHECK-DAG: irretrievable_steps: 0

/*
---
!where {lines: !label l_assign}:
  !value mutex:
    "*": "{}"
...
*/
