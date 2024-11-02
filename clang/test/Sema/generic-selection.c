// RUN: %clang_cc1 -std=c11 -fsyntax-only -Wno-strict-prototypes -Wno-implicit-function-declaration -verify %s
// RUN: %clang_cc1 -std=c99 -pedantic -fsyntax-only -Wno-strict-prototypes -Wno-implicit-function-declaration -verify=expected,ext %s

void g(void);

void foo(int n) {
  (void) _Generic(0, // ext-warning {{'_Generic' is a C11 extension}}
      struct A: 0, // expected-error {{type 'struct A' in generic association incomplete}}
      void(): 0,   // expected-error {{type 'void ()' in generic association not an object type}}
      int[n]: 0);  // expected-error {{type 'int[n]' in generic association is a variably modified type}}

  (void) _Generic(0, // ext-warning {{'_Generic' is a C11 extension}}
      void (*)():     0,  // expected-note {{compatible type 'void (*)()' specified here}}
      void (*)(void): 0); // expected-error {{type 'void (*)(void)' in generic association compatible with previously specified type 'void (*)()'}}

  (void) _Generic((void (*)()) 0, // expected-error {{controlling expression type 'void (*)()' compatible with 2 generic association types}} \
                                  // ext-warning {{'_Generic' is a C11 extension}}
      void (*)(int):  0,  // expected-note {{compatible type 'void (*)(int)' specified here}}
      void (*)(void): 0); // expected-note {{compatible type 'void (*)(void)' specified here}}

  (void) _Generic(0, // expected-error {{controlling expression type 'int' not compatible with any generic association type}} \
                     // ext-warning {{'_Generic' is a C11 extension}}
      char: 0, short: 0, long: 0);

  int a1[_Generic(0, int: 1, short: 2, float: 3, default: 4) == 1 ? 1 : -1]; // ext-warning {{'_Generic' is a C11 extension}}
  int a2[_Generic(0, default: 1, short: 2, float: 3, int: 4) == 4 ? 1 : -1]; // ext-warning {{'_Generic' is a C11 extension}}
  int a3[_Generic(0L, int: 1, short: 2, float: 3, default: 4) == 4 ? 1 : -1]; // ext-warning {{'_Generic' is a C11 extension}}
  int a4[_Generic(0L, default: 1, short: 2, float: 3, int: 4) == 1 ? 1 : -1]; // ext-warning {{'_Generic' is a C11 extension}}
  int a5[_Generic(0, int: 1, short: 2, float: 3) == 1 ? 1 : -1]; // ext-warning {{'_Generic' is a C11 extension}}
  int a6[_Generic(0, short: 1, float: 2, int: 3) == 3 ? 1 : -1]; // ext-warning {{'_Generic' is a C11 extension}}

  int a7[_Generic("test", char *: 1, default: 2) == 1 ? 1 : -1]; // ext-warning {{'_Generic' is a C11 extension}}
  int a8[_Generic(g, void (*)(void): 1, default: 2) == 1 ? 1 : -1]; // ext-warning {{'_Generic' is a C11 extension}}

  const int i = 12;
  int a9[_Generic(i, int: 1, default: 2) == 1 ? 1 : -1]; // ext-warning {{'_Generic' is a C11 extension}}

  // This is expected to not trigger any diagnostics because the controlling
  // expression is not evaluated.
  (void)_Generic(*(int *)0, int: 1); // ext-warning {{'_Generic' is a C11 extension}}
}

int __attribute__((overloadable)) test (int);
double __attribute__((overloadable)) test (double);
char testc(char);

void PR30201(void) {
  _Generic(4, char:testc, default:test)(4); // ext-warning {{'_Generic' is a C11 extension}}
}

void GH50227(void) {
  // Previously, the controlling expression for the outer _Generic makes it
  // result dependent, and testing whether that controlling expression has side
  // effects would cause a crash.
  _Generic( // ext-warning {{'_Generic' is a C11 extension}}
    n(
      _Generic(n++, int : 0) // expected-error {{cannot increment value of type 'int ()'}} ext-warning {{'_Generic' is a C11 extension}}
    ), int : 0);
}

struct Test {
  int i;
};

void unreachable_associations(const int i, const struct Test t) {
  _Static_assert( // ext-warning {{'_Static_assert' is a C11 extension}}
    _Generic(i, // ext-warning {{'_Generic' is a C11 extension}}
      const int : 1,    // expected-warning {{due to lvalue conversion of the controlling expression, association of type 'const int' will never be selected because it is qualified}}
      volatile int : 2, // expected-warning {{due to lvalue conversion of the controlling expression, association of type 'volatile int' will never be selected because it is qualified}}
      int[12] : 3,      // expected-warning {{due to lvalue conversion of the controlling expression, association of type 'int[12]' will never be selected because it is of array type}}
      int : 4,
      default : 5
    ) == 4, "we had better pick int!");
  _Static_assert( // ext-warning {{'_Static_assert' is a C11 extension}}
    _Generic(t, // ext-warning {{'_Generic' is a C11 extension}}
      struct Test : 1,
      const struct Test : 2, // expected-warning {{due to lvalue conversion of the controlling expression, association of type 'const struct Test' will never be selected because it is qualified}}
      default : 3
    ) == 1, "we had better pick struct Test, not const struct Test!"); // C-specific result
}

void GH55562(void) {
  // Ensure that you can still define a type within a generic selection
  // association (despite it not being particularly useful).
  (void)_Generic(1, struct S { int a; } : 0, default : 0); // ext-warning {{'_Generic' is a C11 extension}}
  struct S s = { 0 };
  int i = s.a;
}
