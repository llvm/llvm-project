// RUN: %clang_analyze_cc1 -x c -analyzer-checker=core -analyzer-output=text -verify %s

#define NULL 0

int test_noparammacro(void) {
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

int coin(void);

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
#define DEREF_IN_MACRO(X) int fn(void) {int *p = 0; return *p; }

DEREF_IN_MACRO(0) // expected-warning{{Dereference of null pointer}}
                  // expected-note@-1{{'p' initialized to a null}}
                  // expected-note@-2{{Dereference of null pointer}}

// Warning should not be suppressed if the null returned by the macro
// is not related to the warning.
#define RETURN_NULL() (0)
extern int* returnFreshPointer(void);
int noSuppressMacroUnrelated(void) {
  int *x = RETURN_NULL();
  x = returnFreshPointer();  // expected-note{{Value assigned to 'x'}}
  if (x) {} // expected-note{{Taking false branch}}
            // expected-note@-1{{Assuming 'x' is null}}
  return *x; // expected-warning{{Dereference of null pointer}}
             // expected-note@-1{{Dereference}}
}

// Value haven't changed by the assignment, but the null pointer
// did not come from the macro.
int noSuppressMacroUnrelatedOtherReason(void) {
  int *x = RETURN_NULL();
  x = returnFreshPointer();  
  x = 0; // expected-note{{Null pointer value stored to 'x'}}
  return *x; // expected-warning{{Dereference of null pointer}}
             // expected-note@-1{{Dereference}}
}
