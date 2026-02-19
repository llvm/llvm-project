// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-library -x hlsl -o - -fsyntax-only %s -verify

//===----------------------------------------------------------------------===//
// Baseline: struct with direct incomplete array
//===----------------------------------------------------------------------===//
struct S {
  int a[];
};

export void fn(int A) {
  // expected-error@+1{{too few initializers in list for type 'S' (expected 1 but found 0)}}
  S s = {};
}

//===----------------------------------------------------------------------===//
// Multidimensional arrays with at least one incomplete dimension
//===----------------------------------------------------------------------===//
export void fn_multi_arrays() {
  // Incomplete outer dimension
  // expected-error@+1{{too few initializers in list for type 'int[][2]' (expected 2 but found 0)}}
  int a[][2] = {};

  // Incomplete middle dimension
  // expected-error@+1{{array has incomplete element type 'int[][3]'}}
  int b[2][][3] = {};

  // Incomplete inner dimension
  // expected-error@+1{{array has incomplete element type 'int[]'}}
  int c[2][3][] = {};
}

//===----------------------------------------------------------------------===//
// Struct containing multidimensional incomplete arrays
//===----------------------------------------------------------------------===//
struct S2 {
  int m[][4];
};

export void fn_struct_multi() {
  // expected-error@+1{{too few initializers in list for type 'S2' (expected 1 but found 0)}}
  S2 s = {};
}

//===----------------------------------------------------------------------===//
// Nested structs with incomplete arrays
//===----------------------------------------------------------------------===//
struct Inner {
  int x[];
};

struct Outer {
  Inner I;
};

export void fn_nested_struct() {
  // expected-error@+1{{too few initializers in list for type 'Outer' (expected 1 but found 0)}}
  Outer o = {};
}

//===----------------------------------------------------------------------===//
// Base-class inheritance containing incomplete arrays
//===----------------------------------------------------------------------===//
struct Base {
  int b[];
};

// expected-error@+1{{base class 'Base' has a flexible array member}}
struct Derived : Base {
  int d;
};

export void fn_derived() {
  // expected-error@+1{{too few initializers in list for type 'Derived' (expected 1 but found 0)}}
  Derived d = {};
}

//===----------------------------------------------------------------------===//
// Deep inheritance chain with incomplete array in base
//===----------------------------------------------------------------------===//
struct Base2 {
  int x[];
};

// expected-error@+1{{base class 'Base2' has a flexible array member}}
struct Mid : Base2 {
  int y;
};

struct Final : Mid {
  int z;
};

export void fn_deep_inheritance() {
  // expected-error@+1{{too few initializers in list for type 'Final' (expected 2 but found 0)}}
  Final f = {};
}
