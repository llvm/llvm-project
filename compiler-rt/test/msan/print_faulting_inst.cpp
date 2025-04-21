// Try parameter '0' (program runs cleanly)
// -------------------------------------------------------
// RUN: env MSAN_OPTIONS=print_faulting_instruction=true %clangxx_msan -g %s -o %t && env MSAN_OPTIONS=print_faulting_instruction=true %run %t 0

// Try parameter '1'
// -------------------------------------------------------
// RUN: %clangxx_msan -g %s -o %t && not env MSAN_OPTIONS=print_faulting_instruction=true %run %t 1 >%t.out 2>&1
// RUN: FileCheck --check-prefix STORE-CHECK %s < %t.out

// RUN: %clangxx_msan -mllvm -msan-embed-faulting-instruction=1 -g %s -o %t && not env MSAN_OPTIONS=print_faulting_instruction=true %run %t 1 >%t.out 2>&1
// RUN: FileCheck --check-prefixes VERBOSE-STORE-CHECK,STORE-CHECK %s < %t.out

// RUN: %clangxx_msan -mllvm -msan-embed-faulting-instruction=2 -g %s -o %t && not env MSAN_OPTIONS=print_faulting_instruction=true %run %t 1 >%t.out 2>&1
// RUN: FileCheck --check-prefixes VERY-VERBOSE-STORE-CHECK,STORE-CHECK %s < %t.out

// Try parameter '2', with -fsanitize-memory-param-retval
// -------------------------------------------------------
// RUN: %clangxx_msan -fsanitize-memory-param-retval -g %s -o %t && not env MSAN_OPTIONS=print_faulting_instruction=true %run %t 2 >%t.out 2>&1
// RUN: FileCheck --check-prefix PARAM-CHECK %s < %t.out

// RUN: %clangxx_msan -fsanitize-memory-param-retval -mllvm -msan-embed-faulting-instruction=1 -g %s -o %t && not env MSAN_OPTIONS=print_faulting_instruction=true %run %t 2 >%t.out 2>&1
// RUN: FileCheck --check-prefixes VERBOSE-PARAM-CHECK,PARAM-CHECK %s < %t.out

// RUN: %clangxx_msan -fsanitize-memory-param-retval -mllvm -msan-embed-faulting-instruction=2 -g %s -o %t && not env MSAN_OPTIONS=print_faulting_instruction=true %run %t 2 >%t.out 2>&1
// RUN: FileCheck --check-prefixes VERY-VERBOSE-PARAM-CHECK,PARAM-CHECK %s < %t.out

// Try parameter '2', with -fno-sanitize-memory-param-retval
// -------------------------------------------------------
// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -g %s -o %t && not env MSAN_OPTIONS=print_faulting_instruction=true %run %t 2 >%t.out 2>&1
// RUN: FileCheck --check-prefix NO-PARAM-CHECK %s < %t.out

// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -mllvm -msan-embed-faulting-instruction=1 -g %s -o %t && not env MSAN_OPTIONS=print_faulting_instruction=true %run %t 2 >%t.out 2>&1
// RUN: FileCheck --check-prefixes VERBOSE-NO-PARAM-CHECK,NO-PARAM-CHECK %s < %t.out

// RUN: %clangxx_msan -fno-sanitize-memory-param-retval -mllvm -msan-embed-faulting-instruction=2 -g %s -o %t && not env MSAN_OPTIONS=print_faulting_instruction=true %run %t 2 >%t.out 2>&1
// RUN: FileCheck --check-prefixes VERY-VERBOSE-NO-PARAM-CHECK,NO-PARAM-CHECK %s < %t.out

#include <stdio.h>
#include <stdlib.h>

#define THRICE(o,t) twice(o,t)

__attribute__((noinline)) extern "C" int twice(int o, int t) {
  return o + t < 3;
}

int main(int argc, char *argv[]) {
  int buf[100];
  buf[0] = 42;
  buf[1] = 43;

  if (argc != 2) {
    printf("Usage: %s index\n", argv[0]);
    return 1;
  }

  int index = atoi(argv[1]);
  int val = buf[index];

  printf("index %d, abs(val) %d, THRICE(val,5) %d\n", index, abs(val), THRICE(val,5));
  // VERY-VERBOSE-PARAM-CHECK: Instruction that failed the shadow check: %{{.*}} = call noundef i32 @twice(i32 noundef %{{.*}}, i32 noundef 5)
  // VERBOSE-PARAM-CHECK: Instruction that failed the shadow check: call twice
  // PARAM-CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // PARAM-CHECK: {{#0 0x.* in main .*print_faulting_inst.cpp:}}[[@LINE-4]]

  if (val)
    // VERY-VERBOSE-NO-PARAM-CHECK: Instruction that failed the shadow check:  br i1 %{{.*}}, label %{{.*}}, label %{{.*}}
    // VERBOSE-NO-PARAM-CHECK: Instruction that failed the shadow check: br
    // NO-PARAM-CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
    // NO-PARAM-CHECK: {{#0 0x.* in main .*print_faulting_inst.cpp:}}[[@LINE-4]]
    printf("Variable is non-zero\n");
  else
    printf("Variable is zero\n");

  int nextval = buf[index + 1];
  buf[nextval + abs(index)] = twice(index,6);
  // VERY-VERBOSE-STORE-CHECK: Instruction that failed the shadow check: store i32 %{{.*}}, ptr %{{.*}}
  // VERBOSE-STORE-CHECK: Instruction that failed the shadow check: store
  // STORE-CHECK: WARNING: MemorySanitizer: use-of-uninitialized-value
  // STORE-CHECK: {{#0 0x.* in main .*print_faulting_inst.cpp:}}[[@LINE-4]]

  return 0;
}
