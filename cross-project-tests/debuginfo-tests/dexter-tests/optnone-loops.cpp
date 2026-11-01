// Purpose:
// Verifies that the debugging experience of loops marked optnone is as expected.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// UNSUPPORTED: system-darwin

// RUN: %clang++ -std=gnu++11 -O2 -g %s -o %t
// RUN: %dexter -w \
// RUN:     --binary %t %dexter_lldb_args -- %s | FileCheck %s

// A simple loop of assignments.
// With optimization level > 0 the compiler reorders basic blocks
// based on the basic block frequency analysis information.
// This also happens with optnone and it shouldn't.
// This is not affecting debug info so it is a minor limitation.
// Basic block placement based on the block frequency analysis
// is normally done to improve i-Cache performances.
__attribute__((optnone)) void simple_memcpy_loop(int *dest, const int *src,
                                                 unsigned nelems) {
  for (unsigned i = 0; i != nelems; ++i)
    dest[i] = src[i]; // !dex_label simple_memcpy_loop
}

// A trivial loop that could be optimized into a builtin memcpy
// which is either expanded into a optimal sequence of mov
// instructions or directly into a call to memset@plt
__attribute__((optnone)) void trivial_memcpy_loop(int *dest, const int *src) {
  for (unsigned i = 0; i != 16; ++i)
    dest[i] = src[i]; // !dex_label trivial_memcpy_loop
}

__attribute__((always_inline)) int foo(int a) { return a + 5; }

// A trivial loop of calls to a 'always_inline' function.
__attribute__((optnone)) void nonleaf_function_with_loop(int *dest,
                                                         const int *src) {
  for (unsigned i = 0; i != 16; ++i)
    dest[i] = foo(src[i]); // !dex_label nonleaf_function_with_loop
}

// This entire function could be optimized into a
// simple movl %esi, %eax.
// That is because we can compute the loop trip count
// knowing that ind-var 'i' can never be negative.
__attribute__((optnone)) int counting_loop(unsigned values) {
  unsigned i = 0;
  while (values--) // !dex_label counting_loop
    i++;
  return i;
}

// This loop could be rotated.
// while(cond){
//   ..
//   cond--;
// }
//
//  -->
// if(cond) {
//   do {
//     ...
//     cond--;
//   } while(cond);
// }
//
// the compiler will not try to optimize this function.
// However the Machine BB Placement Pass will try
// to reorder the basic block that computes the
// expression 'count' in order to simplify the control
// flow.
__attribute__((optnone)) int loop_rotate_test(int *src, unsigned count) {
  int result = 0;

  while (count) {
    result += src[count - 1]; // !dex_label loop_rotate_test
    count--;
  }
  return result; // !dex_label loop_rotate_test_ret
}

typedef int *intptr __attribute__((aligned(16)));

// This loop can be vectorized if we enable
// the loop vectorizer.
__attribute__((optnone)) void loop_vectorize_test(intptr dest, intptr src) {
  unsigned count = 0;

  int tempArray[16];

  while (count != 16) { // !dex_label loop_vectorize_test
    tempArray[count] = src[count];
    tempArray[count + 1] = src[count + 1];  // !dex_label loop_vectorize_test_2
    tempArray[count + 2] = src[count + 2];  // !dex_label loop_vectorize_test_3
    tempArray[count + 3] = src[count + 3];  // !dex_label loop_vectorize_test_4
    dest[count] = tempArray[count];         // !dex_label loop_vectorize_test_5
    dest[count + 1] = tempArray[count + 1]; // !dex_label loop_vectorize_test_6
    dest[count + 2] = tempArray[count + 2]; // !dex_label loop_vectorize_test_7
    dest[count + 3] = tempArray[count + 3]; // !dex_label loop_vectorize_test_8
    count += 4;                             // !dex_label loop_vectorize_test_9
  }
}

int main() {
  int A[] = {3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  int B[] = {13, 14, 15, 16, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int C[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};

  simple_memcpy_loop(C, A, 16);
  trivial_memcpy_loop(B, C);
  nonleaf_function_with_loop(B, B);
  int count = counting_loop(16);
  count += loop_rotate_test(B, 16);
  loop_vectorize_test(A, B);

  return A[0] + count;
}

// CHECK-DAG: seen_values: 30
// CHECK-DAG: correct_step_coverage: 100.0%

/*
---
!where {function: simple_memcpy_loop}:
  ? !and
    lines: !label simple_memcpy_loop
    conditions: "i == 0 || i == 4 || i == 8"
  : !value nelems: 16
    !value "src[i]": [3, 7, 1]
!where {function: trivial_memcpy_loop}:
  ? !and
    lines: !label trivial_memcpy_loop
    conditions: "i == 3 || i == 7 || i == 9 || i == 14 || i == 15"
  : !value i: [3, 7, 9, 14, 15]
    !value "dest[i-1] == src[i-1]": "true"
!where {function: nonleaf_function_with_loop}:
  !and {lines: !label nonleaf_function_with_loop, conditions: "i == 1"}:
    !value "dest[0]": 8
    !value "dest[1]": 4
    !value "dest[2]": 5
    !value "src[0]": 8
    !value "src[1]": 4
    !value "src[2]": 5
    !value "src[1] == dest[1]": "true"
    !value "src[2] == dest[2]": "true"
!where {function: counting_loop}:
  !and {lines: !label counting_loop, conditions: "i == 8 || i == 16"}:
    !value i: [8, 16]
!where {function: loop_rotate_test}:
  !and {lines: !label loop_rotate_test, conditions: "result == 13"}:
    !value "src[count]": 13
  !and {lines: !label loop_rotate_test_ret, conditions: "result == 158"}:
    !value result: 158
!where {function: loop_vectorize_test}:
  ? !and
    lines: !range [!label loop_vectorize_test, !label loop_vectorize_test_9]
    conditions: "count == 4 || count == 8 || count == 12 || count == 16"
  : !and {lines: !label loop_vectorize_test_2}:
      !value 'tempArray[count] == src[count]': "true"
    !and {lines: !label loop_vectorize_test_3}:
      !value 'tempArray[count+1] == src[count+1]': "true"
    !and {lines: !label loop_vectorize_test_4}:
      !value 'tempArray[count+2] == src[count+2]': "true"
    !and {lines: !label loop_vectorize_test_5}:
      !value 'tempArray[count+3] == src[count+3]': "true"
    !and {lines: !label loop_vectorize_test_6}:
      !value 'dest[count] == tempArray[count]': "true"
    !and {lines: !label loop_vectorize_test_7}:
      !value 'dest[count+1] == tempArray[count+1]': "true"
    !and {lines: !label loop_vectorize_test_8}:
      !value 'dest[count+2] == tempArray[count+2]': "true"
    !and {lines: !label loop_vectorize_test_9}:
      !value 'dest[count+3] == tempArray[count+3]': "true"
...
*/
