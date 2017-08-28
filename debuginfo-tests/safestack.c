// RUN: %clang %target_itanium_abi_host_triple -arch x86_64 %s -o %t.out -g -fsanitize=safe-stack
// RUN: %test_debuginfo %s %t.out
// REQUIRES: not_asan
//           Zorg configures the ASAN stage2 bots to not build the
//           safestack compiler-rt.  Only run this test on
//           non-asanified configurations.

struct S {
  int a[8];
};

int f(struct S s, unsigned i);

int main(int argc, const char **argv) {
  struct S s = {{0, 1, 2, 3, 4, 5, 6, 7}};
  // DEBUGGER: break 17
  f(s, 4);
  // DEBUGGER: break 19
  return 0;
}

int f(struct S s, unsigned i) {
  // DEBUGGER: break 24
  return s.a[i];
}

// DEBUGGER: r
// DEBUGGER: p s
// CHECK: a = ([0] = 0, [1] = 1, [2] = 2, [3] = 3, [4] = 4, [5] = 5, [6] = 6, [7] = 7)
// DEBUGGER: c
// DEBUGGER: p s
// CHECK: a = ([0] = 0, [1] = 1, [2] = 2, [3] = 3, [4] = 4, [5] = 5, [6] = 6, [7] = 7)
// DEBUGGER: c
// DEBUGGER: p s
// CHECK: a = ([0] = 0, [1] = 1, [2] = 2, [3] = 3, [4] = 4, [5] = 5, [6] = 6, [7] = 7)
