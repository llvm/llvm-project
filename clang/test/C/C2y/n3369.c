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

int global_array[12];

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

void test_zero_size_arrays() {
  int array[0]; // expected-warning {{zero size arrays are an extension}}
  static_assert(_Countof(array) == 0);
  static_assert(_Countof(int[0]) == 0); // expected-warning {{zero size arrays are an extension}}
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
