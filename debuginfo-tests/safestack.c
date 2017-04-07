// RUN: %clang %target_itanium_abi_host_triple -arch x86_64 %s -o %t.out -g -fsanitize=safe-stack
// RUN: %test_debuginfo %s %t.out
//
// DEBUGGER: break 15
// DEBUGGER: r
// DEBUGGER: p s
//
// CHECK: a = ([0] = 0, [1] = 1, [2] = 2, [3] = 3, [4] = 4, [5] = 5, [6] = 6, [7] = 7)

struct S {
  int a[8];
};

int f(struct S s, unsigned i) {
  return s.a[i];
}

int main(int argc, const char **argv) {
  struct S s = {{0, 1, 2, 3, 4, 5, 6, 7}};
  return f(s, 4);
}
