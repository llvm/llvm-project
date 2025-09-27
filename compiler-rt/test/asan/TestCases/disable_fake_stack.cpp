// RUN: %clangxx_asan %s -o %t && %run %t

#include "defines.h"

#include <sanitizer/asan_interface.h>

volatile char *saved;

ATTRIBUTE_NOINLINE bool IsOnStack() {
  volatile char temp = ' ';
  void *fake_stack = __asan_get_current_fake_stack();
  void *real = __asan_addr_is_in_fake_stack(
      fake_stack, const_cast<char *>(&temp), nullptr, nullptr);
  saved = &temp;
  return real == nullptr;
}

int main(int argc, char *argv[]) {
  __asan_disable_fake_stack();
  if (!IsOnStack()) {
    return 1;
  }
  __asan_enable_fake_stack();
  if (IsOnStack()) {
    return 2;
  }
  return 0;
}
