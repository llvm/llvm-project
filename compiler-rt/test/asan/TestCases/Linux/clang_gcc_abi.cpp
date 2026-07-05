// RUN: %clangxx_asan -O0 -x c %s -o %t && not %env_asan_opts=fast_unwind_on_malloc=1 %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O1 -x c %s -o %t && not %env_asan_opts=fast_unwind_on_malloc=1 %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O2 -x c %s -o %t && not %env_asan_opts=fast_unwind_on_malloc=1 %run %t 2>&1 | FileCheck %s
// RUN: %clangxx_asan -O3 -x c %s -o %t && not %env_asan_opts=fast_unwind_on_malloc=1 %run %t 2>&1 | FileCheck %s

// REQUIRES: (arm-target-arch || armhf-target-arch), fast-unwinder-works

#include <stdlib.h>

__attribute__((noinline))
int boom() {
  volatile int three = 3;
  char * volatile s = (char *)malloc(three);
// CHECK: #1 0x{{.*}} in boom {{.*}}clang_gcc_abi.cpp:[[@LINE-1]]
  return s[three]; //BOOM
}

__attribute__((naked, noinline)) void gcc_abi() {
// CHECK: #2 0x{{.*}} in gcc_abi {{.*}}clang_gcc_abi.cpp:[[@LINE+1]]
  asm volatile("str fp, [sp, #-8]!\n\t"
               "str lr, [sp, #4]\n\t"
               "add fp, sp, #4\n\t"
               "bl  boom\n\t"
               "sub sp, fp, #4\n\t"
               "ldr fp, [sp]\n\t"
               "add sp, sp, #4\n\t"
               "ldr pc, [sp], #4\n\t"
              );
}

__attribute__((naked, noinline)) void clang_abi() {
// CHECK: #3 0x{{.*}} in clang_abi {{.*}}clang_gcc_abi.cpp:[[@LINE+1]]
  asm volatile("push {r11, lr}\n\t"
               "mov r11, sp\n\t"
               "bl  gcc_abi\n\t"
               "add r0, r0, #1\n\t"
               "pop {r11, pc}\n\t"
              );
}

int main() {
  clang_abi();
// CHECK: #4 0x{{.*}} in main {{.*}}clang_gcc_abi.cpp:[[@LINE-1]]
}
