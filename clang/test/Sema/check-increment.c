// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify %s -fexperimental-new-constant-interpreter
// expected-no-diagnostics

int printf(const char *, ...);
typedef int *pint;
int main(void) {
   int a[5] = {0};
   pint p = a;
   p++;
   printf("%d\n", *p);
}
