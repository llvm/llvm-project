/* This test case tests that when static allocation for value
 * profiler is on, no malloc/calloc calls will be invoked by
 * profile runtime library. */
#include <stdlib.h>
__attribute__((noinline)) void foo() {}
__attribute__((noinline)) void foo2() {}
void (*FP)();
int MainEntered = 0;
int CallocCalled = 0;
int MallocCalled = 0;

extern void *__real_calloc(size_t s, size_t n);
extern void *__real_malloc(size_t s);

void *__wrap_calloc(size_t s, size_t n) {
  if (MainEntered)
    CallocCalled = 1;
  return __real_calloc(s, n);
}
void *__wrap_malloc(size_t s) {
  if (MainEntered)
    MallocCalled = 1;
  return __real_malloc(s);
}

void getFP(int i) {
  if (i % 2)
    FP = foo;
  else
    FP = foo2;
}

int main() {
  int i;
  MainEntered = 1;
  for (i = 0; i < 100; i++) {
    getFP(i);
    FP();
  }
  return CallocCalled + MallocCalled;
}
