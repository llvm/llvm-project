// Purpose:
// Verifies that the debugging experience of loops marked optnone is as expected.

// REQUIRES: lldb
// UNSUPPORTED: system-windows
// UNSUPPORTED: system-darwin

// RUN: %dexter --fail-lt 1.0 -w \
// RUN:     --builder 'clang' --debugger 'lldb' \
// RUN:     --cflags "-O2 -g" -- %s

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
    dest[i] = src[i]; // DexLabel('target_simple_memcpy_loop')
}

// DexLimitSteps('i', 0, 4, 8, on_line='target_simple_memcpy_loop')
// DexExpectWatchValue('nelems', '16', on_line='target_simple_memcpy_loop')
// DexExpectWatchValue('src[i]', '3', '7', '1', on_line='target_simple_memcpy_loop')


// A trivial loop that could be optimized into a builtin memcpy
// which is either expanded into a optimal sequence of mov
// instructions or directly into a call to memset@plt
__attribute__((optnone)) void trivial_memcpy_loop(int *dest, const int *src) {
  for (unsigned i = 0; i != 16; ++i)
    dest[i] = src[i]; // DexLabel('target_trivial_memcpy_loop')
}

// DexLimitSteps('i', 3, 7, 9, 14, 15, on_line='target_trivial_memcpy_loop')
// DexExpectWatchValue('i', 3, 7, 9, 14, 15, on_line='target_trivial_memcpy_loop')
// DexExpectWatchValue('dest[i-1] == src[i-1]', 'true', on_line='target_trivial_memcpy_loop')


__attribute__((always_inline)) int foo(int a) { return a + 5; }

// A trivial loop of calls to a 'always_inline' function.
__attribute__((optnone)) void nonleaf_function_with_loop(int *dest,
                                                         const int *src) {
  for (unsigned i = 0; i != 16; ++i)
    dest[i] = foo(src[i]); // DexLabel('target_nonleaf_function_with_loop')
}

// DexLimitSteps('i', 1, on_line='target_nonleaf_function_with_loop')
// DexExpectWatchValue('dest[0]', '8', on_line='target_nonleaf_function_with_loop')
// DexExpectWatchValue('dest[1]', '4', on_line='target_nonleaf_function_with_loop')
// DexExpectWatchValue('dest[2]', '5', on_line='target_nonleaf_function_with_loop')
// DexExpectWatchValue('src[0]', '8', on_line='target_nonleaf_function_with_loop')
// DexExpectWatchValue('src[1]', '4', on_line='target_nonleaf_function_with_loop')
// DexExpectWatchValue('src[2]', '5', on_line='target_nonleaf_function_with_loop')

// DexExpectWatchValue('src[1] == dest[1]', 'true', on_line='target_nonleaf_function_with_loop')
// DexExpectWatchValue('src[2] == dest[2]', 'true', on_line='target_nonleaf_function_with_loop')


// This entire function could be optimized into a
// simple movl %esi, %eax.
// That is because we can compute the loop trip count
// knowing that ind-var 'i' can never be negative.
__attribute__((optnone)) int counting_loop(unsigned values) {
  unsigned i = 0;
  while (values--) // DexLabel('target_counting_loop')
    i++;
  return i;
}

// DexLimitSteps('i', 8, 16, on_line='target_counting_loop')
// DexExpectWatchValue('i', 8, 16, on_line='target_counting_loop')


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
    result += src[count - 1]; // DexLabel('target_loop_rotate_test')
    count--;
  }
  return result; // DexLabel('target_loop_rotate_test_ret')
}

// DexLimitSteps('result', 13, on_line='target_loop_rotate_test')
// DexExpectWatchValue('src[count]', 13, on_line='target_loop_rotate_test')
// DexLimitSteps('result', 158, on_line='target_loop_rotate_test_ret')
// DexExpectWatchValue('result', 158, on_line='target_loop_rotate_test_ret')


typedef int *intptr __attribute__((aligned(16)));

// This loop can be vectorized if we enable
// the loop vectorizer.
__attribute__((optnone)) void loop_vectorize_test(intptr dest, intptr src) {
  unsigned count = 0;

  int tempArray[16];

  while(count != 16) { // DexLabel('target_loop_vectorize_test')
    tempArray[count] = src[count];
    tempArray[count+1] = src[count+1]; // DexLabel('target_loop_vectorize_test_2')
    tempArray[count+2] = src[count+2]; // DexLabel('target_loop_vectorize_test_3')
    tempArray[count+3] = src[count+3]; // DexLabel('target_loop_vectorize_test_4')
    dest[count] = tempArray[count]; // DexLabel('target_loop_vectorize_test_5')
    dest[count+1] = tempArray[count+1]; // DexLabel('target_loop_vectorize_test_6')
    dest[count+2] = tempArray[count+2]; // DexLabel('target_loop_vectorize_test_7')
    dest[count+3] = tempArray[count+3]; // DexLabel('target_loop_vectorize_test_8')
    count += 4; // DexLabel('target_loop_vectorize_test_9')
  }
}

// DexLimitSteps('count', 4, 8, 12, 16, from_line='target_loop_vectorize_test', to_line='target_loop_vectorize_test_9')
// DexExpectWatchValue('tempArray[count] == src[count]', 'true', on_line='target_loop_vectorize_test_2')
// DexExpectWatchValue('tempArray[count+1] == src[count+1]', 'true', on_line='target_loop_vectorize_test_3')
// DexExpectWatchValue('tempArray[count+2] == src[count+2]', 'true', on_line='target_loop_vectorize_test_4')
// DexExpectWatchValue('tempArray[count+3] == src[count+3]', 'true', on_line='target_loop_vectorize_test_5')
// DexExpectWatchValue('dest[count] == tempArray[count]', 'true', on_line='target_loop_vectorize_test_6')
// DexExpectWatchValue('dest[count+1] == tempArray[count+1]', 'true', on_line='target_loop_vectorize_test_7')
// DexExpectWatchValue('dest[count+2] == tempArray[count+2]', 'true', on_line='target_loop_vectorize_test_8')
// DexExpectWatchValue('dest[count+3] == tempArray[count+3]', 'true', on_line='target_loop_vectorize_test_9')


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

