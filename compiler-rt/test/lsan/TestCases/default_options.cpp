// RUN: %clangxx_lsan -O2 %s -o %t && %run %t 2>&1 | FileCheck %s

// Fixme: remove once test passes with hwasan
// UNSUPPORTED: hwasan

extern "C"
const char *__lsan_default_options() {
  // CHECK: Available flags for {{Leak|Address}}Sanitizer:
  return "verbosity=1 help=1";
}

int main() {
  return 0;
}
