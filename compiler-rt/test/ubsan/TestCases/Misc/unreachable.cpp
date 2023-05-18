// RUN: %clang %S/Inputs/returns-unexpectedly.c -O3 -c -o %t.ru.o
// RUN: %clangxx -fsanitize=unreachable -O3 -o %t %s %t.ru.o
// RUN: not %run %t builtin 2>&1 | FileCheck %s -check-prefix=BUILTIN
// RUN: not %run %t noreturn-callee-marked 2>&1 | FileCheck %s -check-prefix=NORETURN1
// RUN: not %run %t noreturn-caller-marked 2>&1 | FileCheck %s -check-prefix=NORETURN2

#include <string.h>

void __attribute__((noreturn)) callee_marked_noreturn() {
  // NORETURN1: unreachable.cpp:[[@LINE+1]]:1: runtime error: execution reached an unreachable program point
}

extern "C" void __attribute__((noreturn)) returns_unexpectedly();

int main(int, char **argv) {
  if (strcmp(argv[1], "builtin") == 0)
    // BUILTIN: unreachable.cpp:[[@LINE+1]]:5: runtime error: execution reached an unreachable program point
    __builtin_unreachable();
  else if (strcmp(argv[1], "noreturn-callee-marked") == 0)
    callee_marked_noreturn();
  else if (strcmp(argv[1], "noreturn-caller-marked") == 0)
    // NORETURN2: unreachable.cpp:[[@LINE+1]]:5: runtime error: execution reached an unreachable program point
    returns_unexpectedly();
  return 0;
}
