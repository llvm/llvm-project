// RUN: %clang_cc1 -fsyntax-only -verify %s

int foo(int x) {
  return x == x; // expected-warning {{self-comparison always evaluates to true}}
}

int foo2(int x) {
  return (x) != (((x))); // expected-warning {{self-comparison always evaluates to false}}
}

void foo3(short s, short t) { 
  if (s == s) {} // expected-warning {{self-comparison always evaluates to true}}
  if (s == t) {} // no-warning
}

void foo4(void* v, void* w) {
  if (v == v) {} // expected-warning {{self-comparison always evaluates to true}}
  if (v == w) {} // no-warning
}

int qux(int x) {
   return x < x; // expected-warning {{self-comparison}}
}

int qux2(int x) {
   return x > x; // expected-warning {{self-comparison}}
}

int bar(float x) {
  return x == x; // no-warning
}

int bar2(float x) {
  return x != x; // no-warning
}

#define IS_THE_ANSWER(x) (x == 42)

int macro_comparison() {
  return IS_THE_ANSWER(42);
}

// Don't complain in unevaluated contexts.
int compare_sizeof(int x) {
  return sizeof(x == x); // no-warning
}

int array_comparisons() {
  int array1[2];
  int array2[2];

  //
  // compare same array
  //
  return array1 == array1; // expected-warning{{self-comparison always evaluates to true}}
  return array1 != array1; // expected-warning{{self-comparison always evaluates to false}}
  return array1 < array1; // expected-warning{{self-comparison always evaluates to false}}
  return array1 <= array1; // expected-warning{{self-comparison always evaluates to true}}
  return array1 > array1; // expected-warning{{self-comparison always evaluates to false}}
  return array1 >= array1; // expected-warning{{self-comparison always evaluates to true}}

  //
  // compare different arrays
  //
  return array1 == array2; // expected-warning{{array comparison always evaluates to false}}
  return array1 != array2; // expected-warning{{array comparison always evaluates to true}}

  //
  // we don't know what these are going to be
  //
  return array1 < array2; // expected-warning{{array comparison always evaluates to a constant}}
  return array1 <= array2; // expected-warning{{array comparison always evaluates to a constant}}
  return array1 > array2; // expected-warning{{array comparison always evaluates to a constant}}
  return array1 >= array2; // expected-warning{{array comparison always evaluates to a constant}}

}

// Don't issue a warning when either the left or right side of the comparison
// results from a macro expansion.  <rdar://problem/8435950>
#define R8435950_A i 
#define R8435950_B i 

int R8435950(int i) {
  if (R8435950_A == R8435950_B) // no-warning
   return 0;
  return 1;
}

__attribute__((weak)) int weak_1[3];
__attribute__((weak)) int weak_2[3];
_Bool compare_weak() {
  return weak_1 == weak_2;
}
