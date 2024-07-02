// UNSUPPORTED: target={{.*windows.*}}
// This test is derived from "compiler-rt/test/profile/instrprof-write-buffer-internal.c",
// and that test was disabled on Windows due to sanitizer-windows bot failures.
// Doing the same here. See 2fcc3f4b18.
//
// RUN: rm -f %t.buf.profraw
// RUN: %clang_pgogen -o %t %s
// RUN: %run %t %t.buf.profraw
// RUN: llvm-profdata show %t.buf.profraw | FileCheck %s

// CHECK: Instrumentation level: IR  entry_first = 0
// CHECK: Total functions: 2
// CHECK: Maximum function count: 0
// CHECK: Maximum internal block count: 0

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void threadFunc(void* callback) // Needed for failure.
{
  typedef void (FuncPtr)();
  (*(FuncPtr*)callback)();
}

uint64_t __llvm_profile_get_size_for_buffer();
int __llvm_profile_write_buffer(char*);

int main(int argc, const char *argv[])
{
  // Write to a buffer, and write that to a file.
  uint64_t bufsize = __llvm_profile_get_size_for_buffer();
  char *buf = malloc(bufsize);
  int ret = __llvm_profile_write_buffer(buf);

  if (ret != 0) {
    fprintf(stderr, "failed to write buffer");
    return ret;
  }

  FILE *f = fopen(argv[1], "w");
  fwrite(buf, bufsize, 1, f);
  fclose(f);
  free(buf);

  return 0;
}
