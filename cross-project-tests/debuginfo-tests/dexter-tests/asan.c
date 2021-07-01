// REQUIRES: !asan, lldb
// UNSUPPORTED: system-windows
//           Zorg configures the ASAN stage2 bots to not build the asan
//           compiler-rt. Only run this test on non-asanified configurations.
//
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder 'clang-c' --debugger 'lldb' \
// RUN:     --cflags "--driver-mode=gcc -O0 -glldb -fblocks -arch x86_64 \
// RUN:     -fsanitize=address" --ldflags="-fsanitize=address" -- %s

struct S {
  int a[8];
};

int f(struct S s, unsigned i) {
  return s.a[i]; // DexLabel('asan')
}

int main(int argc, const char **argv) {
  struct S s = {{0, 1, 2, 3, 4, 5, 6, 7}};
  if (f(s, 4) == 4)
    return f(s, 0);
  return 0;
}

// DexExpectWatchValue('s.a[0]', '0', on_line=ref('asan'))
// DexExpectWatchValue('s.a[1]', '1', on_line=ref('asan'))
// DexExpectWatchValue('s.a[7]', '7', on_line=ref('asan'))

