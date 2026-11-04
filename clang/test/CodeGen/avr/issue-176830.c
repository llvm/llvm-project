// RUN: %clang_cc1 -triple avr -target-cpu atmega328 -emit-llvm < %s | FileCheck %s

// This test case verifies https://github.com/llvm/llvm-project/issues/176830,
// and the test code is gained from 
// https://github.com/avrdudes/avr-libc/blob/main/include/string.h

unsigned int strlen(const char *__s) {
  if (__builtin_constant_p (__builtin_strlen (__s))) {
    return __builtin_strlen (__s);
  } else {
    register const char *__r24 __asm("24") = __s;
    register unsigned int __res __asm("24");
    // CHECK: call addrspace(0) i16 asm "call ${2:x}", "={r24},{r24},i,~{r30},~{r31},~{memory}"(ptr %1, ptr addrspace(1) @strlen)
    __asm ("%~call %x2" : "=r" (__res) : "r" (__r24), "i" (strlen) : "30", "31", "memory");
    return __res;
  }
}
