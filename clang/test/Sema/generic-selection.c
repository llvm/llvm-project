// RUN: %clang_cc1 -std=c1x -fsyntax-only -verify %s

void g(void);

void foo(int n) {
  (void) _Generic(0,
      struct A: 0, // expected-error {{type 'struct A' in generic association incomplete}}
      void(): 0,   // expected-error {{type 'void ()' in generic association not an object type}}
      int[n]: 0);  // expected-error {{type 'int [n]' in generic association is a variably modified type}}

  (void) _Generic(0,
      void (*)():     0,  // expected-note {{compatible type 'void (*)()' specified here}}
      void (*)(void): 0); // expected-error {{type 'void (*)(void)' in generic association compatible with previously specified type 'void (*)()'}}

  (void) _Generic((void (*)()) 0, // expected-error {{controlling expression type 'void (*)()' compatible with 2 generic association types}}
      void (*)(int):  0,  // expected-note {{compatible type 'void (*)(int)' specified here}}
      void (*)(void): 0); // expected-note {{compatible type 'void (*)(void)' specified here}}

  (void) _Generic(0, // expected-error {{controlling expression type 'int' not compatible with any generic association type}}
      char: 0, short: 0, long: 0);

  int a1[_Generic(0, int: 1, short: 2, float: 3, default: 4) == 1 ? 1 : -1];
  int a2[_Generic(0, default: 1, short: 2, float: 3, int: 4) == 4 ? 1 : -1];
  int a3[_Generic(0L, int: 1, short: 2, float: 3, default: 4) == 4 ? 1 : -1];
  int a4[_Generic(0L, default: 1, short: 2, float: 3, int: 4) == 1 ? 1 : -1];
  int a5[_Generic(0, int: 1, short: 2, float: 3) == 1 ? 1 : -1];
  int a6[_Generic(0, short: 1, float: 2, int: 3) == 3 ? 1 : -1];

  int a7[_Generic("test", char *: 1, default: 2) == 1 ? 1 : -1];
  int a8[_Generic(g, void (*)(void): 1, default: 2) == 1 ? 1 : -1];

  const int i = 12;
  int a9[_Generic(i, int: 1, default: 2) == 1 ? 1 : -1];

  // This is expected to not trigger any diagnostics because the controlling
  // expression is not evaluated.
  (void)_Generic(*(int *)0, int: 1);
}
