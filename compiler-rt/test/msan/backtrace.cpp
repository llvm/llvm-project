// RUN: %clangxx_msan -O0 %s -o %t && %run %t

#include <assert.h>
#include <execinfo.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

__attribute__((noinline))
void f() {
  void *buf[10];
  int sz = backtrace(buf, sizeof(buf) / sizeof(*buf));
  assert(sz > 0);
  for (int i = 0; i < sz; ++i)
    if (!buf[i]) {
#if defined(__s390x__)
      // backtrace() may return a bogus trailing NULL on s390x.
      if (i == sz - 1)
        continue;
#endif
      exit(1);
    }
  char **s = backtrace_symbols(buf, sz);
  assert(s != 0);
  for (int i = 0; i < sz; ++i)
    printf("%d\n", (int)strlen(s[i]));
}

int main(void) {
  f();
  return 0;
}
