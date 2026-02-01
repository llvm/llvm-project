// RUN: %clang_cc1       -Wmemset-transposed-args -verify %s
// RUN: %clang_cc1 -xc++ -Wmemset-transposed-args -verify %s

#define memset(...) __builtin_memset(__VA_ARGS__)
#define bzero(x,y) __builtin_memset(x, 0, y)
#define real_bzero(x,y) __builtin_bzero(x,y)

int array[10];
int *ptr;

int main(void) {
  memset(array, sizeof(array), 0); // expected-warning{{'size' argument to memset is '0'; did you mean to transpose the last two arguments?}} expected-note{{parenthesize the third argument to silence}}
  memset(array, sizeof(array), 0xff); // expected-warning{{setting buffer to a 'sizeof' expression; did you mean to transpose the last two arguments?}} expected-note{{cast the second argument to 'int' to silence}} expected-warning{{'memset' will always overflow; destination buffer has size 40, but size argument is 255}}
  memset(array, sizeof(array), '\0'); // expected-warning{{'size' argument to memset is '0'; did you mean to transpose the last two arguments?}} expected-note{{parenthesize the third argument to silence}}
  memset(ptr, sizeof(ptr), 0); // expected-warning{{'size' argument to memset is '0'; did you mean to transpose the last two arguments?}} expected-note{{parenthesize the third argument to silence}}
  memset(ptr, sizeof(*ptr) * 10, 1); // expected-warning{{setting buffer to a 'sizeof' expression; did you mean to transpose the last two arguments?}} expected-note{{cast the second argument to 'int' to silence}}
  memset(ptr, sizeof(ptr), '\0');    // expected-warning{{'size' argument to memset is '0'; did you mean to transpose the last two arguments?}} expected-note{{parenthesize the third argument to silence}}
  memset(ptr, 10 * sizeof(int *), 1); // expected-warning{{setting buffer to a 'sizeof' expression; did you mean to transpose the last two arguments?}} expected-note{{cast the second argument to 'int' to silence}}
  memset(ptr, 10 * sizeof(int *) + 10, 0xff); // expected-warning{{setting buffer to a 'sizeof' expression; did you mean to transpose the last two arguments?}} expected-note{{cast the second argument to 'int' to silence}}
  memset(ptr, sizeof(char) * sizeof(int *), 0xff); // expected-warning{{setting buffer to a 'sizeof' expression; did you mean to transpose the last two arguments?}} expected-note{{cast the second argument to 'int' to silence}}
  memset(array, sizeof(array), sizeof(array)); // Uh... fine I guess.
  memset(array, 0, sizeof(array));
  memset(ptr, 0, sizeof(int *) * 10);
  memset(array, (int)sizeof(array), (0)); // no warning
  memset(array, (int)sizeof(array), 32); // no warning
  memset(array, 32, (0)); // no warning
  memset(array, 0, 0); // no warning

  bzero(ptr, 0); // expected-warning{{'size' argument to bzero is '0'}} expected-note{{parenthesize the second argument to silence}}
  real_bzero(ptr, 0); // expected-warning{{'size' argument to bzero is '0'}} expected-note{{parenthesize the second argument to silence}}
}

void macros(void) {
#define ZERO 0
  int array[10];
  memset(array, 0xff, ZERO); // no warning
  // Still emit a diagnostic for memsetting a sizeof expression:
  memset(array, sizeof(array), ZERO); // expected-warning{{'sizeof'}} expected-note{{cast}}
  bzero(array, ZERO); // no warning
  real_bzero(array, ZERO); // no warning
#define NESTED_DONT_DIAG                        \
  memset(array, 0xff, ZERO);                    \
  real_bzero(array, ZERO);

  NESTED_DONT_DIAG;

#define NESTED_DO_DIAG                          \
  memset(array, 0xff, 0);                       \
  real_bzero(array, 0)

  NESTED_DO_DIAG; // expected-warning{{'size' argument to memset}} expected-warning{{'size' argument to bzero}} expected-note2{{parenthesize}}

#define FN_MACRO(p)                             \
  memset(array, 0xff, p)

  FN_MACRO(ZERO);
  FN_MACRO(0); // FIXME: should we diagnose this?

  __builtin_memset(array, 0, ZERO); // no warning
  __builtin_bzero(array, ZERO);
  __builtin_memset(array, 1, 0); // expected-warning{{'size' argument to memset}} // expected-note{{parenthesize}}
  __builtin_bzero(array, 0); // expected-warning{{'size' argument to bzero}} // expected-note{{parenthesize}}
}
