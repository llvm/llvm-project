// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' \
// RUN:     --cflags "-O2 -g" -- %s
// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' \
// RUN:     --cflags "-O0 -g" -- %s

// REQUIRES: lldb
// UNSUPPORTED: system-windows

//// Check that the debugging experience with __attribute__((optnone)) at O2
//// matches O0. Test simple functions performing simple arithmetic
//// operations and small loops.

__attribute__((optnone))
int test1(int test1_a, int test1_b) {
  int test1_result = 0;
  // DexLabel('test1_start')
  test1_result = test1_a + test1_b; // DexExpectStepOrder(1)
  return test1_result; // DexExpectStepOrder(2)
  // DexLabel('test1_end')
}
// DexExpectWatchValue('test1_a', 3, from_line=ref('test1_start'), to_line=ref('test1_end'))
// DexExpectWatchValue('test1_b', 4, from_line=ref('test1_start'), to_line=ref('test1_end'))
// DexExpectWatchValue('test1_result', 0, 7, from_line=ref('test1_start'), to_line=ref('test1_end'))

__attribute__((optnone))
int test2(int test2_a, int test2_b) {
  int test2_result = test2_a + test2_a + test2_a + test2_a;  // DexExpectStepOrder(3)
  // DexLabel('test2_start')
  return test2_a << 2;   // DexExpectStepOrder(4)
  // DexLabel('test2_end')
}
// DexExpectWatchValue('test2_a', 1, from_line=ref('test2_start'), to_line=ref('test2_end'))
// DexExpectWatchValue('test2_b', 2, from_line=ref('test2_start'), to_line=ref('test2_end'))
// DexExpectWatchValue('test2_result', 4, from_line=ref('test2_start'), to_line=ref('test2_end'))

__attribute__((optnone))
int test3(int test3_a, int test3_b) {
  int test3_temp1 = 0, test3_temp2 = 0;
  // DexLabel('test3_start')
  test3_temp1 = test3_a + 5;   // DexExpectStepOrder(5)
  test3_temp2 = test3_b + 5;   // DexExpectStepOrder(6)
  if (test3_temp1 > test3_temp2) { // DexExpectStepOrder(7)
    test3_temp1 *= test3_temp2;    // DexUnreachable()
  }
  return test3_temp1; // DexExpectStepOrder(8)
  // DexLabel('test3_end')
}
// DexExpectWatchValue('test3_a', 5, from_line=ref('test3_start'), to_line=ref('test3_end'))
// DexExpectWatchValue('test3_b', 6, from_line=ref('test3_start'), to_line=ref('test3_end'))
// DexExpectWatchValue('test3_temp1', 0, 10, from_line=ref('test3_start'), to_line=ref('test3_end'))
// DexExpectWatchValue('test3_temp2', 0, 11, from_line=ref('test3_start'), to_line=ref('test3_end'))

unsigned num_iterations = 4;

__attribute__((optnone))
int test4(int test4_a, int test4_b) {
  int val1 = 0, val2 = 0;
  // DexLabel('test4_start')

  val1 = (test4_a > test4_b) ? test4_a : test4_b; // DexExpectStepOrder(9)
  val2 = val1;
  val2 += val1; // DexExpectStepOrder(10)

  for (unsigned i=0; i != num_iterations; ++i) { // DexExpectStepOrder(11, 13, 15, 17, 19)
    val1--;
    val2 += i;
    if (val2 % 2 == 0) // DexExpectStepOrder(12, 14, 16, 18)
      val2 /= 2;
  }

  return (val1 > val2) ? val2 : val1; // DexExpectStepOrder(20)
  // DexLabel('test4_end')
}
// DexExpectWatchValue('test4_a', 1, from_line=ref('test4_start'), to_line=ref('test4_end'))
// DexExpectWatchValue('test4_b', 9, from_line=ref('test4_start'), to_line=ref('test4_end'))
// DexExpectWatchValue('val1', 0, 9, 8, 7, 6, 5, from_line=ref('test4_start'), to_line=ref('test4_end'))
// DexExpectWatchValue('val2', 0, 9, 18, 9, 10, 5, 7, 10, 5, 9, from_line=ref('test4_start'), to_line=ref('test4_end'))

__attribute__((optnone))
int test5(int test5_val) {
  int c = 1;      // DexExpectStepOrder(21)
  // DexLabel('test5_start')
  if (test5_val)  // DexExpectStepOrder(22)
    c = 5;        // DexExpectStepOrder(23)
  return c ? test5_val : test5_val; // DexExpectStepOrder(24)
  // DexLabel('test5_end')
}
// DexExpectWatchValue('test5_val', 7, from_line=ref('test5_start'), to_line=ref('test5_end'))
// DexExpectWatchValue('c', 1, 5, from_line=ref('test5_start'), to_line=ref('test5_end'))

int main() {
  int main_result = 0;
  // DexLabel('main_start')
  main_result = test1(3,4);
  main_result += test2(1,2);
  main_result += test3(5,6);
  main_result += test4(1,9);
  main_result += test5(7);
  return main_result;
  // DexLabel('main_end')
}
// DexExpectWatchValue('main_result', 0, 7, 11, 21, 26, 33, from_line=ref('main_start'), to_line=ref('main_end'))
