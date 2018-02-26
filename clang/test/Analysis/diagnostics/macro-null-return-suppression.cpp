// RUN: %clang_analyze_cc1 -x c -analyzer-checker=core -analyzer-output=text -verify %s

#define NULL 0

int test_noparammacro() {
  int *x = NULL; // expected-note{{'x' initialized to a null pointer value}}
  return *x; // expected-warning{{Dereference of null pointer (loaded from variable 'x')}}
             // expected-note@-1{{Dereference of null pointer (loaded from variable 'x')}}
}

#define DYN_CAST(X) (X ? (char*)X : 0)
#define GENERATE_NUMBER(X) (0)

char test_assignment(int *param) {
  char *param2;
  param2 = DYN_CAST(param);
  return *param2;
}

char test_declaration(int *param) {
  char *param2 = DYN_CAST(param);
  return *param2;
}

int coin();

int test_multi_decl(int *paramA, int *paramB) {
  char *param1 = DYN_CAST(paramA), *param2 = DYN_CAST(paramB);
  if (coin())
    return *param1;
  return *param2;
}

int testDivision(int a) {
  int divider = GENERATE_NUMBER(2); // expected-note{{'divider' initialized to 0}}
  return 1/divider; // expected-warning{{Division by zero}}
                    // expected-note@-1{{Division by zero}}
}

// Warning should not be suppressed if it happens in the same macro.
#define DEREF_IN_MACRO(X) int fn() {int *p = 0; return *p; }

DEREF_IN_MACRO(0) // expected-warning{{Dereference of null pointer}}
                  // expected-note@-1{{'p' initialized to a null}}
                  // expected-note@-2{{Dereference of null pointer}}
