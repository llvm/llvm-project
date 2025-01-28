// A simple fork results in two processes writing to the same file
// RUN: rm -fr %t.profdir
// RUN: %clang_pgogen=%t.profdir -o %t -O2 %s
// RUN: %run %t
// RUN: llvm-profdata show --all-functions --counts %t.profdir/default_*.profraw  | FileCheck %s
//
// CHECK: func1:
// CHECK: Block counts: [4]
// CHECK:  func2:
// CHECK: Block counts: [1]

#include <sys/wait.h>
#include <unistd.h>

__attribute__((noinline)) void func1() {}
__attribute__((noinline)) void func2() {}

int main(void) {
  //                       child     | parent
  int status;         // func1 func2 | func1 func2
  func1();            //   +1        |   +1        (*)
  pid_t pid = fork(); //             |
  if (pid == -1)      //             |
    return 1;         //             |
  if (pid == 0)       //             |
    func2();          //         +1  |
  func1();            //   +1        |   +1
  if (pid)            // ------------+------------
    wait(&status);    //    2     1  |    2    0
  return 0;           // (*)  the child inherits counter values prior to fork
                      //      from the parent in non-continuous mode.
}
