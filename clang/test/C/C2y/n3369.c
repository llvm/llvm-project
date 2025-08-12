// RUN: %clang_cc1 -fsyntax-only -std=c2y -pedantic -Wall -Wno-comment -verify %s
// RUN: %clang_cc1 -fsyntax-only -std=c2y -pedantic -Wall -Wno-comment -fexperimental-new-constant-interpreter -verify %s

/* WG14 N3369: Clang 21
 * _Lengthof operator
 *
 * Adds an operator to get the length of an array. Note that WG14 N3469 renamed
 * this operator to _Countof.
 */

#if !__has_feature(c_countof)
#error "Expected to have _Countof support"
#endif

#if !__has_extension(c_countof)
// __has_extension returns true if __has_feature returns true.
#error "Expected to have _Countof support"
#endif

#define NULL  ((void *) 0)

int global_array[12];
int global_multi_array[12][34];
int global_num;

void test_parsing_failures() {
  (void)_Countof;     // expected-error {{expected expression}}
  (void)_Countof(;    // expected-error {{expected expression}}
  (void)_Countof();   // expected-error {{expected expression}}
  (void)_Countof int; // expected-error {{expected expression}}
}

void test_semantic_failures() {
  (void)_Countof(1);         // expected-error {{'_Countof' requires an argument of array type; 'int' invalid}}
  int non_array;
  (void)_Countof non_array;  // expected-error {{'_Countof' requires an argument of array type; 'int' invalid}}  
  (void)_Countof(int);       // expected-error {{'_Countof' requires an argument of array type; 'int' invalid}}  
  (void)_Countof(test_semantic_failures); // expected-error {{invalid application of '_Countof' to a function type}}
  (void)_Countof(struct S);  // expected-error {{invalid application of '_Countof' to an incomplete type 'struct S'}} \
                                expected-note {{forward declaration of 'struct S'}}
  struct T { int x; };
  (void)_Countof(struct T);  // expected-error {{'_Countof' requires an argument of array type; 'struct T' invalid}}
  struct U { int x[3]; };
  (void)_Countof(struct U);  // expected-error {{'_Countof' requires an argument of array type; 'struct U' invalid}}
  int a[3];
  (void)_Countof(&a);  // expected-error {{'_Countof' requires an argument of array type; 'int (*)[3]' invalid}}
  int *p;
  (void)_Countof(p);  // expected-error {{'_Countof' requires an argument of array type; 'int *' invalid}}
}

void test_constant_expression_behavior(int n) {
  static_assert(_Countof(global_array) == 12);
  static_assert(_Countof global_array == 12);  
  static_assert(_Countof(int[12]) == 12);

  // Use of a VLA makes it not a constant expression, same as with sizeof.
  int array[n];
  static_assert(_Countof(array)); // expected-error {{static assertion expression is not an integral constant expression}}
  static_assert(sizeof(array));   // expected-error {{static assertion expression is not an integral constant expression}}
  static_assert(_Countof(int[n]));// expected-error {{static assertion expression is not an integral constant expression}}
  static_assert(sizeof(int[n]));  // expected-error {{static assertion expression is not an integral constant expression}}
  
  // Constant folding works the same way as sizeof, too.
  const int m = 12;
  int other_array[m];
  static_assert(sizeof(other_array));   // expected-error {{static assertion expression is not an integral constant expression}}
  static_assert(_Countof(other_array)); // expected-error {{static assertion expression is not an integral constant expression}}
  static_assert(sizeof(int[m]));        // expected-error {{static assertion expression is not an integral constant expression}}
  static_assert(_Countof(int[m]));      // expected-error {{static assertion expression is not an integral constant expression}}
  
  // Note that this applies to each array dimension.
  int another_array[n][7];
  static_assert(_Countof(another_array)); // expected-error {{static assertion expression is not an integral constant expression}}
  static_assert(_Countof(*another_array) == 7);

  // Only the first dimension is needed for constant evaluation; other
  // dimensions can be ignored.
  int yet_another_array[7][n];
  static_assert(_Countof(yet_another_array) == 7);
  static_assert(_Countof(*yet_another_array)); // expected-error {{static assertion expression is not an integral constant expression}}
  
  int one_more_time[n][n][7];
  static_assert(_Countof(one_more_time));  // expected-error {{static assertion expression is not an integral constant expression}}
  static_assert(_Countof(*one_more_time)); // expected-error {{static assertion expression is not an integral constant expression}}
  static_assert(_Countof(**one_more_time) == 7);
}

void test_with_function_param(int array[12], int (*array_ptr)[12], int static_array[static 12]) {
  (void)_Countof(array); // expected-error {{'_Countof' requires an argument of array type; 'int *' invalid}}
  static_assert(_Countof(*array_ptr) == 12);
  (void)_Countof(static_array); // expected-error {{'_Countof' requires an argument of array type; 'int *' invalid}}
}

void test_func_fix_fix(int i, char (*a)[3][5], int (*x)[_Countof(*a)], char (*)[_Generic(x, int (*)[3]: 1)]);  // expected-note {{passing argument to parameter}}
void test_func_fix_var(int i, char (*a)[3][i], int (*x)[_Countof(*a)], char (*)[_Generic(x, int (*)[3]: 1)]);  // expected-note {{passing argument to parameter}}
void test_func_fix_uns(int i, char (*a)[3][*], int (*x)[_Countof(*a)], char (*)[_Generic(x, int (*)[3]: 1)]);  // expected-note {{passing argument to parameter}}

void test_funcs() {
  int i3[3];
  int i5[5];
  char c35[3][5];
  test_func_fix_fix(5, &c35, &i3, NULL);
  test_func_fix_fix(5, &c35, &i5, NULL); // expected-warning {{incompatible pointer types passing 'int (*)[5]' to parameter of type 'int (*)[3]'}}
  test_func_fix_var(5, &c35, &i3, NULL);
  test_func_fix_var(5, &c35, &i5, NULL); // expected-warning {{incompatible pointer types passing 'int (*)[5]' to parameter of type 'int (*)[3]'}}
  test_func_fix_uns(5, &c35, &i3, NULL);
  test_func_fix_uns(5, &c35, &i5, NULL); // expected-warning {{incompatible pointer types passing 'int (*)[5]' to parameter of type 'int (*)[3]'}}
}

void test_multidimensional_arrays() {
  int array[12][7];
  static_assert(_Countof(array) == 12);
  static_assert(_Countof(*array) == 7);

  int mdarray[12][7][100][3];
  static_assert(_Countof(mdarray) == 12);
  static_assert(_Countof(*mdarray) == 7);
  static_assert(_Countof(**mdarray) == 100);
  static_assert(_Countof(***mdarray) == 3);
}

void test_unspecified_array_length() {
  static_assert(_Countof(int[])); // expected-error {{invalid application of '_Countof' to an incomplete type 'int[]'}}

  extern int x[][6][3];
  static_assert(_Countof(x)); // expected-error {{invalid application of '_Countof' to an incomplete type 'int[][6][3]'}}
  static_assert(_Countof(*x) == 6);
  static_assert(_Countof(**x) == 3);
}

void test_completed_array() {
  int a[] = {1, 2, global_num};
  static_assert(_Countof(a) == 3);
}

// Test that the return type of _Countof is what you'd expect (size_t).
void test_return_type() {
  static_assert(_Generic(typeof(_Countof global_array), typeof(sizeof(0)) : 1, default : 0));
}

// Test that _Countof is able to look through typedefs.
void test_typedefs() {
  typedef int foo[12];
  foo f;
  static_assert(_Countof(foo) == 12);
  static_assert(_Countof(f) == 12);

  // Ensure multidimensional arrays also work.
  foo x[100];
  static_assert(_Generic(typeof(x), int[100][12] : 1, default : 0));
  static_assert(_Countof(x) == 100);
  static_assert(_Countof(*x) == 12);
}

void test_zero_size_arrays(int n) {
  int array[0]; // expected-warning {{zero size arrays are an extension}}
  static_assert(_Countof(array) == 0);
  static_assert(_Countof(int[0]) == 0); // expected-warning {{zero size arrays are an extension}}
  int multi_array[0][n]; // FIXME: Should trigger -Wzero-length-array
  static_assert(_Countof(multi_array) == 0);
  int another_one[0][3]; // expected-warning {{zero size arrays are an extension}}
  static_assert(_Countof(another_one) == 0);
}

void test_struct_members() {
  struct S {
    int array[10];
  } s;
  static_assert(_Countof(s.array) == 10);

  struct T {
    int count;
    int fam[];
  } t;
  static_assert(_Countof(t.fam)); // expected-error {{invalid application of '_Countof' to an incomplete type 'int[]'}}
}

void test_compound_literals() {
  static_assert(_Countof((int[2]){}) == 2);
  static_assert(_Countof((int[]){1, 2, 3, 4}) == 4);	
}

/* We don't get a diagnostic for test_f1(), because it ends up unused
 * as _Countof() results in an integer constant expression, which is not
 * evaluated.  However, test_f2() ends up being evaluated, since 'a' is
 * a VLA.
 */
static int test_f1();
static int test_f2(); // FIXME: Should trigger function 'test_f2' has internal linkage but is not defined

void test_symbols() {
  int a[global_num][global_num];

  static_assert(_Countof(global_multi_array[test_f1()]) == 34);
  (void)_Countof(a[test_f2()]);
}
