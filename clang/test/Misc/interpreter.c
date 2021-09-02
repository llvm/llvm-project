// RUN: clang-interpreter %s | FileCheck %s
// REQUIRES: native, examples

int printf(const char *, ...);

int main() {
  // CHECK: {{Hello world!}}
  printf("Hello world!\n");
  return 0;
}
