// UNSUPPORTED: target={{.*-windows-gnu}}

// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: %run %t

#include <windows.h>

DWORD WINAPI thread_proc_1(void *) {
  volatile int x, y, z;
  x = 1;
  y = 2;
  z = 3;
  return 0;
}

DWORD WINAPI thread_proc_2(void *) {
  volatile char stack_buffer[42];
  for (int i = 0; i < sizeof(stack_buffer); ++i)
    stack_buffer[i] = 42;
  return 0;
}

int main(void) {
  HANDLE thr = NULL;

  thr = CreateThread(NULL, 0, thread_proc_1, NULL, 0, NULL);
  if (thr == 0)
    return 1;
  if (WAIT_OBJECT_0 != WaitForSingleObject(thr, INFINITE))
    return 2;

  thr = CreateThread(NULL, 0, thread_proc_2, NULL, 0, NULL);
  if (thr == 0)
    return 3;
  if (WAIT_OBJECT_0 != WaitForSingleObject(thr, INFINITE))
    return 4;
  CloseHandle(thr);
}

