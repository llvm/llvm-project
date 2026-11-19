// RUN: %clang_cc1 -fsyntax-only -pedantic -verify=c-pedantic %s
// RUN: %clang_cc1 -fsyntax-only -verify=cxx -x c++ %s

void f(void) {
  _Atomic(int (*)(void)) fp;
  fp++; // c-pedantic-warning {{arithmetic on a pointer to the function type 'int (void)' is a GNU extension}} \
        // cxx-error {{arithmetic on a pointer to the function type 'int ()'}}
}
