// RUN: %clang_cl /E -Xclang -frewrite-includes -- %s | %clang_cl /c -Xclang -verify /Tp -
// expected-no-diagnostics

// This test uses dos-style \r\n line endings.
// Make sure your editor doesn't rewrite them to unix-style \n line endings.
int foo();
int bar();
#define HELLO \
  foo(); \
  bar();

int main() {
  HELLO
  return 0;
}
