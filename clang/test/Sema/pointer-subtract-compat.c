// RUN: %clang_cc1 %s -fsyntax-only -verify -pedantic

typedef const char rchar;
int a(char* a, rchar* b) {
  return a-b;
}

void f0(void (*fp)(void)) {
  int x = fp - fp; // expected-warning{{arithmetic on pointers to the function type 'void (void)' is a GNU extension}}
}
