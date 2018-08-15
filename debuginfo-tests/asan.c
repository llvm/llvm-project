// RUN: %clang -fblocks %target_itanium_abi_host_triple -arch x86_64 %s -o %t.out -g -fsanitize=address
// RUN: %test_debuginfo %s %t.out
// REQUIRES: not_asan
//           Zorg configures the ASAN stage2 bots to not build the asan
//           compiler-rt. Only run this test on non-asanified configurations.
//

struct S {
  int a[8];
};

int f(struct S s, unsigned i) {
  // DEBUGGER: break 14
  return s.a[i];
}

int main(int argc, const char **argv) {
  struct S s = {{0, 1, 2, 3, 4, 5, 6, 7}};
  if (f(s, 4) == 4)
    return f(s, 0);
  return 0;
}

// DEBUGGER: r
// DEBUGGER: p s
// CHECK: a =
// DEBUGGER: p s.a[0]
// CHECK: = 0
// DEBUGGER: p s.a[1]
// CHECK: = 1
// DEBUGGER: p s.a[7]
