// RUN: %clang_cl %s -o %t.exe -fuse-ld=lld -Z7
// RUN: grep DE[B]UGGER: %s | sed -e 's/.*DE[B]UGGER: //' > %t.script
// RUN: %cdb -cf %t.script %t.exe | FileCheck %s --check-prefixes=DEBUGGER,CHECK

#include <stdio.h>
int main() {
  printf("hello world\n");
  int x = 42;
  __debugbreak();
  // DEBUGGER: g
  // DEBUGGER: dv
  // CHECK: x = 0n42
}
// DEBUGGER: q
