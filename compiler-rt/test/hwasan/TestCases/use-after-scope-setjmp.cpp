// RUN: %clangxx_hwasan  -mllvm -hwasan-use-stack-safety=0 -mllvm -hwasan-use-after-scope -O2 %s -o %t && \
// RUN:     %run %t 2>&1

// REQUIRES: aarch64-target-arch
// REQUIRES: stable-runtime

#include <sanitizer/hwasan_interface.h>
#include <setjmp.h>
#include <stdlib.h>
#include <string.h>

#include <sys/types.h>
#include <unistd.h>

volatile const char *stackbuf = nullptr;
jmp_buf jbuf;

__attribute__((noinline)) bool jump() {
  // Fool the compiler so it cannot deduce noreturn.
  if (getpid() != 0) {
    longjmp(jbuf, 1);
    return true;
  }
  return false;
}

bool target() {
  switch (setjmp(jbuf)) {
  case 1:
    return false;
  default:
    break;
  }

  while (true) {
    char buf[4096];
    stackbuf = buf;
    if (!jump()) {
      break;
    };
  }
  return true;
}

int main() {
  target();

  void *untagged = __hwasan_tag_pointer(stackbuf, 0);
  if (stackbuf == untagged) {
    // The buffer wasn't tagged in the first place, so the test will not work
    // as expected.
    return 2;
  }
  if (__hwasan_test_shadow(untagged, 4096) != -1) {
    return 1;
  }

  return 0;
}
