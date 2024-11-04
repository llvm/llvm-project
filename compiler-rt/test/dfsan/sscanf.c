// RUN: %clang_dfsan %s -o %t && %run %t
// XFAIL: *

#include <assert.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
  char buf[256] = "10000000000-100000000000 rw-p 00000000 00:00 0";
  long rss = 0;
  // This test exposes a bug in DFSan's sscanf, that leads to flakiness
  // in release_shadow_space.c (see
  // https://github.com/llvm/llvm-project/issues/91287)
  if (sscanf(buf, "Garbage text before, %ld, Garbage text after", &rss) == 1) {
    printf("Error: matched %ld\n", rss);
    return 1;
  }

  return 0;
}
