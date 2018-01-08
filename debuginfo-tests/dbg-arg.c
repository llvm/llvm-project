// This test case checks debug info during register moves for an argument.
// RUN: %clang %target_itanium_abi_host_triple -m64 -mllvm -fast-isel=false  %s -c -o %t.o -g
// RUN: %clang %target_itanium_abi_host_triple -m64 %t.o -o %t.out
// RUN: %test_debuginfo %s %t.out
//
// DEBUGGER: break 26
// DEBUGGER: r
// DEBUGGER: print mutex
// CHECK:  ={{.* 0x[0-9A-Fa-f]+}}
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


int foobar(struct _mtx *mutex) {
  int r = 1;
  int l = 0;
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
