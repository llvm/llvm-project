// Test -fsanitize-coverage=edge,indirect-call,trace-pc
// RUN: %clangxx_asan -O0 -DTRACE_RT %s -o %t-rt.o -c
// RUN: %clangxx_asan -O0 -fsanitize-coverage=edge,trace-pc,indirect-calls %s -o %t %t-rt.o
// RUN: %run %t
#ifdef TRACE_RT
int pc_count;
void *last_callee;
extern "C" void __sanitizer_cov_trace_pc() {
  pc_count++;
}
extern "C" void __sanitizer_cov_trace_pc_indir(void *callee) {
  last_callee = callee;
}
#else
#include <stdio.h>
#include <assert.h>
extern int pc_count;
extern void *last_callee;

__attribute__((noinline)) void foo() { printf("foo\n"); }
__attribute__((noinline)) void bar() { printf("bar\n"); }

int main(int argc, char **argv) {
  void (*f)(void) = argc ? foo : bar;
  int c1 = pc_count;
  f();
  int c2 = pc_count;
  assert(c1 < c2);
  assert(last_callee == foo);
}
#endif
