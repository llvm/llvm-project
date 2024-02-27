// RUN: %clang_dfsan -fno-sanitize=dataflow -O2 -fPIE -DCALLBACKS -c %s -o %t-callbacks.o
// RUN: %clang_dfsan -gmlt -fsanitize-ignorelist=%S/Inputs/flags_abilist.txt -O2 -mllvm -dfsan-reaches-function-callbacks=1 %s %t-callbacks.o -o %t
// RUN: %run %t 2>&1 | FileCheck %s

// RUN: %clang_dfsan -fno-sanitize=dataflow -O2 -fPIE -DCALLBACKS -DORIGIN_TRACKING -c %s -o %t-callbacks.o
// RUN: %clang_dfsan -gmlt -fsanitize-ignorelist=%S/Inputs/flags_abilist.txt -O2 -mllvm -dfsan-reaches-function-callbacks=1 -mllvm -dfsan-track-origins=2 %s %t-callbacks.o -o %t
// RUN: %run %t 2>&1 | FileCheck --check-prefix=CHECK-ORIGIN-TRACKING %s

// Tests that callbacks are inserted for reached functions when
// -dfsan-reaches-function-callbacks is specified.

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <sanitizer/dfsan_interface.h>

#ifdef CALLBACKS
// Compile this code without DFSan to avoid recursive instrumentation.

void my_dfsan_reaches_function_callback(dfsan_label label, dfsan_origin origin,
                                        const char *file, unsigned int line,
                                        const char *function) {
#ifdef ORIGIN_TRACKING
  dfsan_print_origin_id_trace(origin);
#else
  printf("%s:%d %s\n", file, line, function);
#endif
}

#else

__attribute__((noinline)) uint64_t add(uint64_t *a, uint64_t *b) {

  return *a + *b;
  // CHECK: {{.*}}compiler-rt/test/dfsan/reaches_function.c:[[# @LINE - 1]] add.dfsan
  // CHECK-ORIGIN-TRACKING: Origin value: 0x10000002, Taint value was stored to memory at
  // CHECK-ORIGIN-TRACKING: #0 {{.*}} in add.dfsan {{.*}}compiler-rt/test/dfsan/reaches_function.c:[[# @LINE - 3]]:{{.*}}
  // CHECK-ORIGIN-TRACKING: Origin value: 0x1, Taint value was created at
  // CHECK-ORIGIN-TRACKING: #0 {{.*}} in main {{.*}}compiler-rt/test/dfsan/reaches_function.c:{{.*}}
}

extern void my_dfsan_reaches_function_callback(dfsan_label label,
                                               dfsan_origin origin,
                                               const char *file,
                                               unsigned int line,
                                               const char *function);

int main(int argc, char *argv[]) {

  dfsan_set_reaches_function_callback(my_dfsan_reaches_function_callback);

  uint64_t a = 0;
  uint64_t b = 0;

  dfsan_set_label(8, &a, sizeof(a));
  uint64_t c = add(&a, &b);
  // CHECK: {{.*}}compiler-rt/test/dfsan/reaches_function.c:[[# @LINE - 1]] main
  // CHECK-ORIGIN-TRACKING: Origin value: 0x10000002, Taint value was stored to memory at
  // CHECK-ORIGIN-TRACKING: #0 {{.*}} in add.dfsan {{.*}}compiler-rt/test/dfsan/reaches_function.c:{{.*}}
  // CHECK-ORIGIN-TRACKING: Origin value: 0x1, Taint value was created at
  // CHECK-ORIGIN-TRACKING: #0 {{.*}} in main {{.*}}compiler-rt/test/dfsan/reaches_function.c:[[# @LINE - 6]]:{{.*}}
  return c;
}

#endif // #ifdef CALLBACKS
