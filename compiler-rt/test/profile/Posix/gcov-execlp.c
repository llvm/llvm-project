/// A basic block with fork/exec* is split. .gcda is flushed immediately before
/// fork/exec* so the lines before exec* are counted once while succeeding
/// lines are not counted.
// RUN: mkdir -p %t.d && cd %t.d
// RUN: %clang --coverage %s -o %t
// RUN: test -f gcov-execlp.gcno
// RUN: rm -f gcov-execlp.gcda && %run %t
// RUN: llvm-cov gcov -t gcov-execlp.gcda | FileCheck %s --check-prefixes=CHECK,EXECLP

// RUN: %clang --coverage -DEXECVP %s -o %t
// RUN: rm -f gcov-execlp.gcda && %run %t
// RUN: llvm-cov gcov -t gcov-execlp.gcda | FileCheck %s --check-prefixes=CHECK,EXECVP

#include <unistd.h>

void func1(void) {}                // CHECK:          1: [[#@LINE]]:void func1(void)
void func2(void) {}                // CHECK-NEXT: #####: [[#@LINE]]:
int main(void) {                   // CHECK-NEXT:     1: [[#@LINE]]:
  func1();                         // CHECK-NEXT:     1: [[#@LINE]]:
#ifdef EXECVP
  char *argv[] = {"ls", "-l", (char *)0};
  execvp("ls", argv);              // EXECVP:         1: [[#@LINE]]:  execvp
#else
  execlp("ls", "-l", (char *)0); // EXECLP:     1: [[#@LINE]]:  execlp
#endif
  func2();                         // CHECK:      #####: [[#@LINE]]:  func2
  return 0;                        // CHECK-NEXT: #####: [[#@LINE]]:
}
