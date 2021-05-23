// REQUIRES: lldb
// UNSUPPORTED: system-windows
//
// This test case checks debug info during register moves for an argument.
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder clang-c --debugger 'lldb' \
// RUN:     --cflags "-m64 -mllvm -fast-isel=false -g" -- %s
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
  int l = 0; // DexLabel('l_assign')
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


/*
DexExpectProgramState({
  'frames': [
    {
      'location': { 'lineno': ref('l_assign') },
      'watches': {
        '*mutex': { 'is_irretrievable': False }
      }
    }
  ]
})
*/

