// A simple fork results in two processes writing to the same file
// RUN: rm -fr %t.profdir
// RUN: %clang_pgogen=%t.profdir -o %t -O2 %s
// RUN: %run %t
// RUN: llvm-profdata show --all-functions --counts %t.profdir/default_*.profraw  | FileCheck %s
// RUN: rm -fr %t.profdir
// RUN: env LLVM_PROFILE_NO_MMAP=1 %run %t
// RUN: llvm-profdata show --all-functions --counts %t.profdir/default_*.profraw  | FileCheck %s

//
// CHECK: func1:
// CHECK: Block counts: [21]
// CHECK:  func2:
// CHECK: Block counts: [10]

#include <sys/wait.h>
#include <unistd.h>

__attribute__((noinline)) void func1() {}
__attribute__((noinline)) void func2() {}

int main(void) {
  //                           child     | parent
  //                         func1 func2 | func1 func2
  func1();              //   +10       |   +1        (*)
  int i = 10;           //             |
  while (i-- > 0) {     //             |
    pid_t pid = fork(); //             |
    if (pid == -1)      //             |
      return 1;         //             |
    if (pid == 0) {     //             |
      func2();          //         +10 |
      func1();          //   +10       |
      return 0;         //             |
    }                   //             |
  }                     // ------------+------------
  int status;           //   20     10 |   1     0
  i = 10;               // (*)  the child inherits counter values prior to fork
  while (i-- > 0)       //      from the parent in non-continuous mode.
    wait(&status);
  return 0;
}
