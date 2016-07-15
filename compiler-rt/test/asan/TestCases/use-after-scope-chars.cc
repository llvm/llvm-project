// RUN: %clangxx_asan -O1 -fsanitize-address-use-after-scope %s -o %t && \
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
