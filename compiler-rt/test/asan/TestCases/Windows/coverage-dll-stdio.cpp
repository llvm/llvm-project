// UNSUPPORTED: target={{.*-windows-gnu}}

// Test that coverage and MSVC CRT stdio work from a DLL. This ensures that the
// __local_stdio_printf_options function isn't instrumented for coverage.

// RUN: rm -rf %t && mkdir %t && cd %t
// RUN: %clang_cl_asan -fsanitize-coverage=func,trace-pc-guard -Od %p/dll_host.cpp -Fet.exe
// RUN: %clang_cl_asan -fsanitize-coverage=func,trace-pc-guard -LD -Od %s -Fet.dll
// RUN: %run ./t.exe t.dll 2>&1 | FileCheck %s

#include <stdio.h>

extern "C" __declspec(dllexport)
int test_function() {
  printf("hello world\n");
  // CHECK: hello world
  return 0;
}
