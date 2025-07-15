// MSAN re-execs on unlimited stacks. We use that to verify ReExec() uses the
// right path.
// REQUIRES: shell
// RUN: %clangxx_msan -O0 %s -o %t && ulimit -s unlimited && %run %t | FileCheck %s

#include <stdio.h>

#if !defined(__GLIBC_PREREQ)
#define __GLIBC_PREREQ(a, b) 0
#endif

#if __GLIBC_PREREQ(2, 16)
#include <sys/auxv.h>
#endif

int main() {
#if __GLIBC_PREREQ(2, 16)
  // Make sure AT_EXECFN didn't get overwritten by re-exec.
  puts(reinterpret_cast<const char *>(getauxval(AT_EXECFN)));
#else
  puts("No getauxval");
#endif
  // CHECK-NOT: /proc/self/exe
}
