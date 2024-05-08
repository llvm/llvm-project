// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -o /dev/null

int foo(void) {
  register int a __asm__("ebx");
  register int b __asm__("r16");
  return a;
}
