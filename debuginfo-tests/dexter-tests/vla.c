// This test case verifies the debug location for variable-length arrays.
// REQUIRES: lldb
// UNSUPPORTED: system-windows
//
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder clang-c --debugger 'lldb' --cflags "-O0 -glldb" -- %s

void init_vla(int size) {
  int i;
  int vla[size];
  for (i = 0; i < size; i++)
    vla[i] = size-i;
  vla[0] = size; // DexLabel('end_init')
}

int main(int argc, const char **argv) {
  init_vla(23);
  return 0;
}

// DexExpectWatchValue('vla[0]', '23', on_line='end_init')
// DexExpectWatchValue('vla[1]', '22', on_line='end_init')

