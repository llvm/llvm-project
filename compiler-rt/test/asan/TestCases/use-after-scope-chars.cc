// RUN: %clangxx_asan -O1 -mllvm -asan-use-after-scope=1 %s -o %t && \
// RUN:     not %run %t 2>&1 | FileCheck %s
// XFAIL: *

// FIXME: This works only for arraysize <= 8.

char *p = 0;

int main() {
  {
    char x[1024] = {};
    p = x;
  }
  return *p;  // BOOM
}
