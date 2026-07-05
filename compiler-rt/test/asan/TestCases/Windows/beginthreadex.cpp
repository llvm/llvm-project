// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: %run %t

#include <windows.h>
#include <process.h>

unsigned WINAPI thread_proc(void *) {
  volatile char stack_buffer[42];
  for (int i = 0; i < sizeof(stack_buffer); ++i)
    stack_buffer[i] = 42;
  return 0;
}

int main() {
  HANDLE thr = (HANDLE)_beginthreadex(NULL, 0, thread_proc, NULL, 0, NULL);
  if (thr == 0)
    return 1;
  if (WAIT_OBJECT_0 != WaitForSingleObject(thr, INFINITE))
    return 2;
  CloseHandle(thr);
}
