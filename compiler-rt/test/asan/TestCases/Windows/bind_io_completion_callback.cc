// Make sure we can throw exceptions from work items executed via
// BindIoCompletionCallback.
//
// Clang doesn't support exceptions on Windows yet, so for the time being we
// build this program in two parts: the code with exceptions is built with CL,
// the rest is built with Clang.  This represents the typical scenario when we
// build a large project using "clang-cl -fallback -fsanitize=address".
//
// RUN: %clangxx_asan %s -o %t.exe
// RUN: %run %t.exe 2>&1 | FileCheck %s

#include <windows.h>
#include <stdio.h>

void ThrowAndCatch();

__declspec(noinline)
void Throw() {
  fprintf(stderr, "Throw\n");
// CHECK: Throw
  throw 1;
}

void ThrowAndCatch() {
  int local;
  try {
    Throw();
  } catch(...) {
    fprintf(stderr, "Catch\n");
// CHECK: Catch
  }
}

char buffer[65536];
HANDLE done;
OVERLAPPED ov;

void CALLBACK completion_callback(DWORD error, DWORD bytesRead,
                                  LPOVERLAPPED pov) {
  ThrowAndCatch();
  SetEvent(done);
}

int main(int argc, char **argv) {
  done = CreateEvent(0, false, false, "job is done");
  if (!done)
    return 1;
  HANDLE file = CreateFile(
      argv[0], GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE, NULL,
      OPEN_EXISTING,
      FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED,
      NULL);
  if (!file)
    return 2;
  if (!BindIoCompletionCallback(file, completion_callback, 0))
    return 3;

  if (!ReadFile(file, buffer, sizeof(buffer), NULL, &ov) &&
      GetLastError() != ERROR_IO_PENDING)
    return 4;

  if (WAIT_OBJECT_0 != WaitForSingleObject(done, 10 * 1000))
    return 5;
  fprintf(stderr, "Done!\n");
// CHECK: Done!
}
