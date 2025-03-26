// RUN: %clang_cc1 -fsyntax-only -std=c2y -pedantic -Wall -Wno-comment -verify %s

/* WG14 N3369: Clang 21
 * _Lengthof operator
 *
 * Adds an operator to get the length of an array. Note that WG14 N3469 renamed
 * this operator to _Countof.
 */

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
}

void test_with_function_param(int array[12], int (*array_ptr)[12]) {
  (void)_Countof(array); // expected-error {{'_Countof' requires an argument of array type; 'int *' invalid}}
  static_assert(_Countof(*array_ptr) == 12);
}

void test_multidimensional_arrays() {
  int array[12][7];
  static_assert(_Countof(array) == 12);
  static_assert(_Countof(*array) == 7);
}
