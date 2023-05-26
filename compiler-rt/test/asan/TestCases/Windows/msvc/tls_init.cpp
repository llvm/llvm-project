// RUN: %clang_cl_asan %s -Fe%t.exe /MD
// RUN: %run %t.exe | FileCheck %s

// CHECK: my_thread_callback
// CHECK: ran_before_main: 1

#include <windows.h>
#include <stdio.h>
#include <string.h>

#pragma comment (lib, "dbghelp")

static bool ran_before_main = false;

extern "C" void __asan_init(void);

static void NTAPI /*__attribute__((no_sanitize_address))*/
my_thread_callback(PVOID module, DWORD reason, PVOID reserved) {
  ran_before_main = true;
  static const char str[] = "my_thread_callback\n";

  // Fail the test if we aren't called for the expected reason or we can't write
  // stdout.
  if (reason != DLL_PROCESS_ATTACH)
    return;
  HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE);
  if (!out || out == INVALID_HANDLE_VALUE)
    return;

  DWORD written = 0;
  WriteFile(out, &str[0], sizeof(str), &written, NULL);
}

extern "C" {
#pragma const_seg(".CRT$XLC")
extern const PIMAGE_TLS_CALLBACK p_thread_callback;
const PIMAGE_TLS_CALLBACK p_thread_callback = my_thread_callback;
#pragma const_seg()
}

#ifdef _WIN64
#pragma comment(linker, "/INCLUDE:_tls_used")
#pragma comment(linker, "/INCLUDE:p_thread_callback")
#else
#pragma comment(linker, "/INCLUDE:__tls_used")
#pragma comment(linker, "/INCLUDE:_p_thread_callback")
#endif

int main() {
  printf("ran_before_main: %d\n", ran_before_main);
}
