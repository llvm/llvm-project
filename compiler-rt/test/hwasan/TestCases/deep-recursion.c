// RUN: %clang_hwasan -O1 %s -o %t
// RUN: %env_hwasan_opts=stack_history_size=1 not %run %t 2>&1 | FileCheck %s --check-prefix=D1
// RUN: %env_hwasan_opts=stack_history_size=2 not %run %t 2>&1 | FileCheck %s --check-prefix=D2
// RUN: %env_hwasan_opts=stack_history_size=3 not %run %t 2>&1 | FileCheck %s --check-prefix=D3
// RUN: %env_hwasan_opts=stack_history_size=5 not %run %t 2>&1 | FileCheck %s --check-prefix=D5
// RUN:                                       not %run %t 2>&1 | FileCheck %s --check-prefix=DEFAULT

// Run the same tests as above, but using the __hwasan_add_frame_record libcall.
// The output should be the exact same.
// RUN: %clang_hwasan -O1 %s -o %t -mllvm -hwasan-record-stack-history=libcall
// RUN: %env_hwasan_opts=stack_history_size=1 not %run %t 2>&1 | FileCheck %s --check-prefix=D1
// RUN: %env_hwasan_opts=stack_history_size=2 not %run %t 2>&1 | FileCheck %s --check-prefix=D2
// RUN: %env_hwasan_opts=stack_history_size=3 not %run %t 2>&1 | FileCheck %s --check-prefix=D3
// RUN: %env_hwasan_opts=stack_history_size=5 not %run %t 2>&1 | FileCheck %s --check-prefix=D5
// RUN:                                       not %run %t 2>&1 | FileCheck %s --check-prefix=DEFAULT

// Stack histories are currently not recorded on x86.
// XFAIL: target=x86_64{{.*}}

#include <stdlib.h>
// At least -O1 is needed for this function to not have a stack frame on
// AArch64.
void USE(void *x) { // pretend_to_do_something(void *x)
  __asm__ __volatile__("" : : "r" (x) : "memory");
}

volatile int four = 4;

__attribute__((noinline)) void OOB() { int x[4]; x[four] = 0; USE(&x[0]); }
__attribute__((noinline)) void FUNC1() { int x; USE(&x); OOB(); }
__attribute__((noinline)) void FUNC2() { int x; USE(&x); FUNC1(); }
__attribute__((noinline)) void FUNC3() { int x; USE(&x); FUNC2(); }
__attribute__((noinline)) void FUNC4() { int x; USE(&x); FUNC3(); }
__attribute__((noinline)) void FUNC5() { int x; USE(&x); FUNC4(); }
__attribute__((noinline)) void FUNC6() { int x; USE(&x); FUNC5(); }
__attribute__((noinline)) void FUNC7() { int x; USE(&x); FUNC6(); }
__attribute__((noinline)) void FUNC8() { int x; USE(&x); FUNC7(); }
__attribute__((noinline)) void FUNC9() { int x; USE(&x); FUNC8(); }
__attribute__((noinline)) void FUNC10() { int x; USE(&x); FUNC9(); }

int main() { FUNC10(); }

// D1: Previously allocated frames
// D1: in OOB
// D1-NOT: in FUNC
// D1: Memory tags around the buggy address

// D2: Previously allocated frames
// D2: in OOB
// D2: in FUNC1
// D2-NOT: in FUNC
// D2: Memory tags around the buggy address

// D3: Previously allocated frames
// D3: in OOB
// D3: in FUNC1
// D3: in FUNC2
// D3-NOT: in FUNC
// D3: Memory tags around the buggy address

// D5: Previously allocated frames
// D5: in OOB
// D5: in FUNC1
// D5: in FUNC2
// D5: in FUNC3
// D5: in FUNC4
// D5-NOT: in FUNC
// D5: Memory tags around the buggy address

// DEFAULT: Previously allocated frames
// DEFAULT: in OOB
// DEFAULT: in FUNC1
// DEFAULT: in FUNC2
// DEFAULT: in FUNC3
// DEFAULT: in FUNC4
// DEFAULT: in FUNC5
// DEFAULT: in FUNC6
// DEFAULT: in FUNC7
// DEFAULT: in FUNC8
// DEFAULT: in FUNC9
// DEFAULT: in FUNC10
// DEFAULT-NOT: in FUNC
// DEFAULT: Memory tags around the buggy address
