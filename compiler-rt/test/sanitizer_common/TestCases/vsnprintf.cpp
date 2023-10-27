// Test that the common part implementation of *printf interceptors does not
// cause negative-size-param false positives.

// RUN: %clangxx -O2 %s -o %t
// RUN: %env_tool_opts=check_printf=1 %run %t 2>&1

// FIXME: The maximum supported allocation size is too platform-specific:
// REQUIRES: x86_64-target-arch

// FIXME: printf is not intercepted on Windows yet.
// UNSUPPORTED: target={{.*windows-msvc.*}}

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void write(char *buf, int buf_size, const char *fmt, ...) {
  va_list args;
  va_start(args, fmt);
  vsnprintf(buf, buf_size, fmt, args);
  va_end(args);
}

int main() {
  char buffer[100];
  const size_t kStrSize = 1UL << 31;
  char *x = (char *)malloc(kStrSize);
  memset(x, '=', kStrSize - 1);
  x[kStrSize - 1] = 0;
  write(buffer, 100, "%s\n", x);
  free(x);
  return 0;
}
