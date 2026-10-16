// RUN: %clang_cl -MD -c -o %t %s
// RUN: %llvm_jitlink %t 2>&1 | FileCheck %s
// CHECK: Initializer worker completed
// CHECK-NEXT: Entering main

#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

static HANDLE worker_thread;
static int initializer_timed_out;

static void on_exit(void) {}

static DWORD WINAPI worker(void *context) {
  (void)context;
  atexit(on_exit);
  return 0;
}

static void initialize(void) {
  worker_thread = CreateThread(NULL, 0, worker, NULL, 0, NULL);
  if (!worker_thread) {
    initializer_timed_out = 1;
    return;
  }

  if (WaitForSingleObject(worker_thread, 10000) != WAIT_OBJECT_0) {
    initializer_timed_out = 1;
    return;
  }

  puts("Initializer worker completed");
  fflush(stdout);
}

#pragma section(".CRT$XCU", read)
__declspec(allocate(".CRT$XCU")) void (*initializer)(void) = initialize;

int main(int argc, char *argv[]) {
  (void)argc;
  (void)argv;

  if (worker_thread) {
    WaitForSingleObject(worker_thread, INFINITE);
    CloseHandle(worker_thread);
  }
  if (initializer_timed_out)
    return 1;

  puts("Entering main");
  fflush(stdout);
  return 0;
}
