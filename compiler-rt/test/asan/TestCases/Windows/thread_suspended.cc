// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: %run %t

#include <windows.h>

DWORD WINAPI thread_proc(void *) {
  volatile char stack_buffer[42];
  for (int i = 0; i < sizeof(stack_buffer); ++i)
    stack_buffer[i] = 42;
  return 0x42;
}

int main() {
  DWORD exitcode;
  HANDLE thr = CreateThread(NULL, 0, thread_proc, NULL, CREATE_SUSPENDED, NULL);
  ResumeThread(thr);
  if (thr == 0)
    return 1;
  if (WAIT_OBJECT_0 != WaitForSingleObject(thr, INFINITE))
    return 2;

  GetExitCodeThread(thr, &exitcode);
  if (exitcode != 0x42)
    return 3;
  CloseHandle(thr);
}

