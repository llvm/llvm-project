// RUN: %clang -ccc-echo -o %t.exe %s 2> %t.log

// Make sure we used clang.
// RUN: grep 'clang\(-[0-9.]\+\)\?\(\.[Ee][Xx][Ee]\)\?" -cc1 .*hello.c' %t.log

// RUN: %t.exe > %t.out
// RUN: grep "I'm a little driver, short and stout." %t.out

// FIXME: We don't have a usable assembler on Windows, so we can't build real
// apps yet.
// XFAIL: win32

#include <stdio.h>

int main() {
  printf("I'm a little driver, short and stout.");
  return 0;
}
