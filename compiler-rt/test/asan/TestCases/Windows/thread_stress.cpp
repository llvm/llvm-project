// RUN: %clang_cl_asan %Od %s %Fe%t
// RUN: %run %t

#include <windows.h>

DWORD WINAPI thread_proc(void *) {
  volatile char stack_buffer[42];
  for (int i = 0; i < sizeof(stack_buffer); ++i)
    stack_buffer[i] = 42;
  return 0;
}

int main(void) {
  for (int iter = 0; iter < 1024; ++iter) {
    const int NUM_THREADS = 8;
    HANDLE thr[NUM_THREADS];
    for (int i = 0; i < NUM_THREADS; ++i) {
      thr[i] = CreateThread(NULL, 0, thread_proc, NULL, 0, NULL);
      if (thr[i] == 0)
        return 1;
    }
    for (int i = 0; i < NUM_THREADS; ++i) {
      if (WAIT_OBJECT_0 != WaitForSingleObject(thr[i], INFINITE))
        return 2;
      CloseHandle(thr[i]);
    }
  }
  return 0;
}

