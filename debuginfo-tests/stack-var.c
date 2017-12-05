// RUN: %clang %target_itanium_abi_host_triple %s -O -o %t.out -gdwarf-2
// RUN: %test_debuginfo %s %t.out

void __attribute__((noinline, optnone)) bar(int *test) {}
int main() {
  int test;
  test = 23;
  // DEBUGGER: break 12
  // DEBUGGER: r
  // DEBUGGER: p test
  // CHECK: = 23
  bar(&test);
  // DEBUGGER: break 17
  // DEBUGGER: c
  // DEBUGGER: p test
  // CHECK: = 23
  return test;
}
