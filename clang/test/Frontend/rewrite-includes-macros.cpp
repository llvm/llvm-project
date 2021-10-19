// REQUIRES: system-windows
// RUN: %clang_cl /E -Xclang -frewrite-includes %s | %clang_cl /c -Xclang -verify /Tp -
// expected-no-diagnostics

int foo();
int bar();
#define HELLO \
  foo(); \
  bar();

int main() {
  HELLO
  return 0;
}
