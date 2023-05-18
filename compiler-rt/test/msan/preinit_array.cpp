// RUN: %clangxx_msan -O0 %s -o %t && %run %t

// FIXME: Something changed in glibc 2.34, maybe earier.
// UNSUPPORTED: glibc-2.34

#include <sanitizer/msan_interface.h>

volatile int global;
static void pre_ctor() {
  volatile int local;
  global = 42;
  local = 42;
}

__attribute__((section(".preinit_array"), used)) void(*__local_pre_ctor)(void) = pre_ctor;

int main(void) {
  return 0;
}
