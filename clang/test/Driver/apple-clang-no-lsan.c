// Apple-Clang: Don't support LSan
// REQUIRES: system-darwin
// RUN: not %clang -fsanitize=leak %s -o %t 2>&1 | FileCheck %s
// CHECK: unsupported option '-fsanitize=leak'
int main() {
  return 0;
}
