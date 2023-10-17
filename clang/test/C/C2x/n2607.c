// RUN: %clang_cc1 -std=c2x -Wall -pedantic -verify %s
// RUN: %clang_cc1 -std=c17 -Wall -pedantic -verify %s

/* WG14 N2607: Partial
 * Compatibility of Pointers to Arrays with Qualifiers
 *
 * FIXME: We consider this partially implemented because there are still issues
 * with the composite type from a conditional operator. Further, we don't issue
 * any diagnostics in C17 or earlier when we need at least a pedantic
 * diagnostic about the type incompatibilities.
 */

void matrix_fun(int N, const float x[N][N]);
void test1(void) {
  int N = 100;
  float x[N][N];
  // FIXME: This is OK in C23 but should be diagnosed as passing incompatible
  // pointer types in C17 and earlier.
  matrix_fun(N, x);
}

#define TEST(Expr, Type) _Generic(Expr, Type : 1)

void test2(void) {
  typedef int array[1];
  array reg_array;
  const array const_array;

  // An array and its elements are identically qualified. We have to test this
  // using pointers to the array and element, because the controlling
  // expression of _Generic will undergo lvalue conversion, which drops
  // qualifiers.
  (void)_Generic(&reg_array, int (*)[1] : 1);
  (void)_Generic(&reg_array[0], int * : 1);

  (void)_Generic(&const_array, const int (*)[1] : 1);
  (void)_Generic(&const_array[0], const int * : 1);

  // But identical qualification does *not* apply to the _Atomic specifier,
  // because that's a special case. You cannot apply the _Atomic specifier or
  // qualifier to an array type directly.
  _Atomic array atomic_array;       // expected-error {{_Atomic cannot be applied to array type 'array'}}
  _Atomic(array) also_atomic_array; // expected-error {{_Atomic cannot be applied to array type 'array'}}

  // Ensure we support arrays of restrict-qualified pointer types.
  int *restrict array_of_restricted_ptrs[1];
  int *restrict md_array_of_restricted_ptrs[1][1];
}

void test3(void) {
  // Validate that we pick the correct composite type for a conditional
  // operator in the presence of qualifiers.
  const int const_array[1];
  int array[1];

  // FIXME: the type here should be `const int (*)[1]`, but for some reason,
  // Clang is deciding it's `void *`. This relates to N2607 because the
  // conditional operator is not properly implementing 6.5.15p7 regarding
  // qualifiers, despite that wording not being touched by this paper.
  // However, it should get a pedantic diagnostic in C17 about use of
  // incompatible pointer types.
  (void)_Generic(1 ? &const_array : &array, const int (*)[1] : 1);   /* expected-error {{controlling expression type 'void *' not compatible with any generic association type}}
                                                                        expected-warning {{pointer type mismatch ('const int (*)[1]' and 'int (*)[1]')}}
                                                                      */
}
