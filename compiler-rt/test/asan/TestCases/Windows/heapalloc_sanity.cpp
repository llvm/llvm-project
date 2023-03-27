// UNSUPPORTED: target={{.*-windows-gnu}}

// RUN: %clang_cl_asan -Od %s -Fe%t
// RUN: %run %t 2>&1 | FileCheck %s
#include <stdio.h>
#include <windows.h>

int main() {
  char *buffer;
  buffer = (char *)HeapAlloc(GetProcessHeap(), 0, 32),
  buffer[0] = 'a';
  HeapFree(GetProcessHeap(), 0, buffer);
  puts("Okay");
  // CHECK: Okay
}
