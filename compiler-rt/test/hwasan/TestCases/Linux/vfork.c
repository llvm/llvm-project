// https://github.com/google/sanitizers/issues/925
// RUN: %clang_hwasan -O0 %s -o %t && %run %t 2>&1

// REQUIRES: aarch64-target-arch || x86_64-target-arch
// REQUIRES: pointer-tagging

#include <assert.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <stdio.h>
#include <sanitizer/hwasan_interface.h>

__attribute__((noinline, no_sanitize("hwaddress"))) void child() {
  char x[10000];
  __hwasan_tag_memory(x, 0xAA, sizeof(x));
  _exit(0);
}

__attribute__((noinline, no_sanitize("hwaddress"))) void parent() {
  char x[10000];
  __hwasan_print_shadow(&x, sizeof(x));
  assert(__hwasan_test_shadow(x, sizeof(x)) == -1);
}

int main(int argc, char **argv) {
  if (vfork())
    parent();
  else
    child();

  return 0;
}
