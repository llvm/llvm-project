// RUN: %clang++ -std=gnu++11 -O2 -g %s -o %t
// RUN: %dexter -w \
// RUN:     --binary %t %dexter_lldb_args -- %s | FileCheck %s
// RUN: %clang++ -std=gnu++11 -O0 -g %s -o %t
// RUN: %dexter -w \
// RUN:     --binary %t %dexter_lldb_args -- %s | FileCheck %s

// REQUIRES: lldb, D136396
// Currently getting intermittent failures on darwin.
// UNSUPPORTED: system-windows, system-darwin

//// Check that the debugging experience with __attribute__((optnone)) at O2
//// matches O0. Test simple functions performing simple arithmetic
//// operations and small loops.

__attribute__((optnone))
int test1(int test1_a, int test1_b) {
  int test1_result = 0;
  // !dex_label test1_start
  test1_result = test1_a + test1_b;
  return test1_result;
  // !dex_label test1_end
}

__attribute__((optnone))
int test2(int test2_a, int test2_b) {
  int test2_result = test2_a + test2_a + test2_a + test2_a;
  // !dex_label test2_start
  return test2_a << 2;
  // !dex_label test2_end
}

__attribute__((optnone))
int test3(int test3_a, int test3_b) {
  int test3_temp1 = 0, test3_temp2 = 0;
  // !dex_label test3_start
  test3_temp1 = test3_a + 5;
  test3_temp2 = test3_b + 5;
  if (test3_temp1 > test3_temp2) {
    test3_temp1 *= test3_temp2;
  }
  return test3_temp1;
  // !dex_label test3_end
}

unsigned num_iterations = 4;

__attribute__((optnone))
int test4(int test4_a, int test4_b) {
  int val1 = 0, val2 = 0;
  // !dex_label test4_start

  val1 = (test4_a > test4_b) ? test4_a : test4_b;
  val2 = val1;
  val2 += val1;

  for (unsigned i = 0; i != num_iterations; ++i) {
    val1--;
    val2 += i;
    if (val2 % 2 == 0)
      val2 /= 2;
  }

  return (val1 > val2) ? val2 : val1;
  // !dex_label test4_end
}

__attribute__((optnone))
int test5(int test5_val) {
  int c = 1;
  // !dex_label test5_start
  if (test5_val)
    c = 5;
  return c ? test5_val : test5_val;
  // !dex_label test5_end
}

__attribute__((optnone))
int main() {
  int main_result = 0;
  // !dex_label main_start
  main_result = test1(3,4);
  main_result += test2(1,2);
  main_result += test3(5,6);
  main_result += test4(1,9);
  main_result += test5(7);
  return main_result;
  // !dex_label main_end
}

// CHECK-DAG: seen_values: 154
// CHECK-DAG: correct_step_coverage: 100.0%
// CHECK-DAG: correct_line_score: 100.0%

/*
---
!where {function: test1}:
  !and {lines: !range [!label test1_start, !label test1_end]}:
    !value test1_a: 3
    !value test1_b: 4
    !value test1_result: [0, 7]
  !step order: [!label test1_start + 1, !label test1_start + 2]
!where {function: test2}:
  !and {lines: !range [!label test2_start, !label test2_end]}:
    !value test2_a: 1
    !value test2_b: 2
    !value test2_result: 4
  !step order: [!label test2_start - 1, !label test2_start + 1]
!where {function: test3}:
  !and {lines: !range [!label test3_start, !label test3_end]}:
    !value test3_a: 5
    !value test3_b: 6
    !value test3_temp1: [0, 10]
    !value test3_temp2: [0, 11]
  !step order:
    - !label test3_start + 1
    - !label test3_start + 2
    - !label test3_start + 3
    - !label test3_end - 1
  !step never: [!label test3_start + 4]
!where {function: test4}:
  !and {lines: !range [!label test4_start, !label test4_end]}:
    !value test4_a: 1
    !value test4_b: 9
    !value val1: [0, 9, 8, 7, 6, 5]
    !value val2: [0, 9, 18, 9, 10, 5, 7, 10, 5, 9]
  !step order:
    - !label test4_start + 2
    - !label test4_start + 4
    - !label test4_start + 6
    - !label test4_start + 9
    - !label test4_start + 6
    - !label test4_start + 9
    - !label test4_start + 6
    - !label test4_start + 9
    - !label test4_start + 6
    - !label test4_start + 9
    - !label test4_start + 6
    - !label test4_end - 1
!where {function: test5}:
  !and {lines: !range [!label test5_start, !label test5_end]}:
    !value test5_val: 7
    !value c: [1, 5]
  !step order:
    - !label test5_start - 1
    - !label test5_start + 1
    - !label test5_start + 2
    - !label test5_start + 3
!where {function: main}:
  !and {lines: !range [!label main_start, !label main_end]}:
    !value main_result: [0, 7, 11, 21, 26, 33]
...
*/
