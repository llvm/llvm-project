// RUN: %clangxx -fsanitize=realtime %s -o %t
// RUN: env RTSAN_OPTIONS="verbosity=1,asdf=1" %run %t 2>&1 | FileCheck %s
// UNSUPPORTED: ios

// Intent: Make sure we are respecting some basic common flags

int main() {
  return 0;
  // CHECK: WARNING: found 1 unrecognized flag(s):
  // CHECK-NEXT: {{.*asdf*}}
}
