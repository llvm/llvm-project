// RUN: %clang -target s390x-linux-gnu -fstack-clash-protection -### %s 2>&1 | FileCheck %s -check-prefix=SystemZ
// SystemZ: "-fstack-clash-protection"
// RUN: %clang -target s390x-linux-gnu -fstack-clash-protection -S -emit-llvm -o %t.ll %s 2>&1 | FileCheck %s -check-prefix=SystemZ-warn
// SystemZ-warn: warning: Unable to protect inline asm that clobbers stack pointer against stack clash

int foo(int c) {
  int r;
  __asm__("ag %%r15, %0"
          :
          : "rm"(c)
          : "r15");
  return r;
}
