// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: env ASAN_OPTIONS=handle_segv=0 %run %t 2>&1 | FileCheck %s --check-prefix=USER
// RUN: env ASAN_OPTIONS=handle_segv=1 not %run %t 2>&1 | FileCheck %s --check-prefix=ASAN
// Test the default.
// RUN: not %run %t 2>&1 | FileCheck %s --check-prefix=ASAN

// This test exits zero when its unhandled exception filter is set. ASan should
// not disturb it when handle_segv=0.

// USER: in main
// USER: in SEHHandler

// ASAN: in main
// ASAN: ERROR: AddressSanitizer: access-violation

#include <windows.h>
#include <stdio.h>

static long WINAPI SEHHandler(EXCEPTION_POINTERS *info) {
  DWORD exception_code = info->ExceptionRecord->ExceptionCode;
  if (exception_code == EXCEPTION_ACCESS_VIOLATION) {
    fprintf(stderr, "in SEHHandler\n");
    fflush(stderr);
    TerminateProcess(GetCurrentProcess(), 0);
  }
  return EXCEPTION_CONTINUE_SEARCH;
}

int main() {
  SetUnhandledExceptionFilter(SEHHandler);
  fprintf(stderr, "in main\n");
  fflush(stderr);

  volatile int *p = nullptr;
  *p = 42;
}
