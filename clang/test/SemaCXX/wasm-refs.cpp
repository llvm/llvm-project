// RUN: %clang_cc1 -fcxx-exceptions -fsyntax-only -verify -std=gnu++11 -triple wasm32 -Wno-unused-value -target-feature +reference-types %s

// This file tests C++ specific constructs with WebAssembly references and
// tables. See wasm-refs-and-tables.c for C constructs.

__externref_t ref;
__externref_t &ref_ref1 = ref; // expected-error {{reference to WebAssembly reference type is not allowed}}
__externref_t &ref_ref2(ref);  // expected-error {{reference to WebAssembly reference type is not allowed}}

static __externref_t table[0];                    // expected-error {{array has sizeless element type '__externref_t'}}
static __externref_t (&ref_to_table1)[0] = table; // expected-error {{array has sizeless element type '__externref_t'}}
static __externref_t (&ref_to_table2)[0](table);  // expected-error {{array has sizeless element type '__externref_t'}}

void illegal_argument_1(__externref_t &r); // expected-error {{reference to WebAssembly reference type is not allowed}}
void illegal_argument_2(__externref_t (&t)[0]); // expected-error {{array has sizeless element type '__externref_t'}}

__externref_t &illegal_return_1(); // expected-error {{reference to WebAssembly reference type is not allowed}}
__externref_t (&illegal_return_2())[0]; // expected-error {{array has sizeless element type '__externref_t'}}

void illegal_throw1() throw(__externref_t);   // expected-error {{sizeless type '__externref_t' is not allowed in exception specification}}
void illegal_throw2() throw(__externref_t *); // expected-error {{pointer to WebAssembly reference type is not allowed}}
void illegal_throw3() throw(__externref_t &); // expected-error {{reference to WebAssembly reference type is not allowed}}
void illegal_throw4() throw(__externref_t[0]); // expected-error {{array has sizeless element type '__externref_t'}}

class RefClass {
  __externref_t f1;       // expected-error {{field has sizeless type '__externref_t'}}
  __externref_t f2[0];    // expected-error {{array has sizeless element type '__externref_t'}}
  __externref_t f3[];     // expected-error {{array has sizeless element type '__externref_t'}}
  __externref_t f4[0][0]; // expected-error {{array has sizeless element type '__externref_t'}}
  __externref_t *f5;      // expected-error {{pointer to WebAssembly reference type is not allowed}}
  __externref_t ****f6;   // expected-error {{pointer to WebAssembly reference type is not allowed}}
  __externref_t (*f7)[0]; // expected-error {{array has sizeless element type '__externref_t'}}
};

struct AStruct {};

template <typename T>
struct TemplatedStruct {
  T f; // expected-error {{field has sizeless type '__externref_t'}}
  void foo(T);
  T bar(void);
  T arr[0]; // expected-error {{array has sizeless element type '__externref_t'}}
  T *ptr;   // expected-error {{pointer to WebAssembly reference type is not allowed}}
};

void func() {
  int foo = 40;
  static_cast<__externref_t>(foo);      // expected-error {{static_cast from 'int' to '__externref_t' is not allowed}}
  static_cast<__externref_t *>(&foo);   // expected-error {{pointer to WebAssembly reference type is not allowed}}
  static_cast<int>(ref);                // expected-error {{static_cast from '__externref_t' to 'int' is not allowed}}
  __externref_t(10);                    // expected-error {{functional-style cast from 'int' to '__externref_t' is not allowed}}
  int i(ref);                           // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type '__externref_t'}}
  const_cast<__externref_t[0]>(table);  // expected-error {{array has sizeless element type '__externref_t'}}
  const_cast<__externref_t *>(table);   // expected-error {{pointer to WebAssembly reference type is not allowed}}
  reinterpret_cast<__externref_t>(foo); // expected-error {{reinterpret_cast from 'int' to '__externref_t' is not allowed}}
  reinterpret_cast<int>(ref);           // expected-error {{reinterpret_cast from '__externref_t' to 'int' is not allowed}}
  int iarr[0];
  reinterpret_cast<__externref_t[0]>(iarr); // expected-error {{array has sizeless element type '__externref_t'}}
  reinterpret_cast<__externref_t *>(iarr);  // expected-error {{pointer to WebAssembly reference type is not allowed}}
  dynamic_cast<__externref_t>(foo);         // expected-error {{invalid target type '__externref_t' for dynamic_cast; target type must be a reference or pointer type to a defined class}}
  dynamic_cast<__externref_t *>(&foo);      // expected-error {{pointer to WebAssembly reference type is not allowed}}

  TemplatedStruct<__externref_t> ts1;    // expected-note {{in instantiation}}
  TemplatedStruct<__externref_t *> ts2;  // expected-error {{pointer to WebAssembly reference type is not allowed}}
  TemplatedStruct<__externref_t &> ts3;  // expected-error {{reference to WebAssembly reference type is not allowed}}
  TemplatedStruct<__externref_t[0]> ts4; // expected-error {{array has sizeless element type '__externref_t'}}

  auto auto_ref = ref;

  auto fn1 = [](__externref_t x) { return x; };
  auto fn2 = [](__externref_t *x) { return x; };   // expected-error {{pointer to WebAssembly reference type is not allowed}}
  auto fn3 = [](__externref_t &x) { return x; };   // expected-error {{reference to WebAssembly reference type is not allowed}}
  auto fn4 = [](__externref_t x[0]) { return x; }; // expected-error {{array has sizeless element type '__externref_t'}}
  auto fn5 = [&auto_ref](void) { return true; };   // expected-error {{cannot capture WebAssembly reference}}
  auto fn6 = [auto_ref](void) { return true; };    // expected-error {{cannot capture WebAssembly reference}}
  auto fn7 = [&](void) { auto_ref; return true; };                        // expected-error {{cannot capture WebAssembly reference}}
  auto fn8 = [=](void) { auto_ref; return true; };                        // expected-error {{cannot capture WebAssembly reference}}

  alignof(__externref_t);    // expected-error {{invalid application of 'alignof' to sizeless type '__externref_t'}}
  alignof(ref);              // expected-warning {{'alignof' applied to an expression is a GNU extension}} expected-error {{invalid application of 'alignof' to sizeless type '__externref_t'}}
  alignof(__externref_t[0]); // expected-error {{array has sizeless element type '__externref_t'}}

  throw ref;  // expected-error {{cannot throw object of sizeless type '__externref_t'}}
  throw &ref; // expected-error {{cannot take address of WebAssembly reference}}

  try {
  } catch (__externref_t) { // expected-error {{cannot catch sizeless type '__externref_t'}}
  }
  try {
  } catch (__externref_t *) { // expected-error {{pointer to WebAssembly reference type is not allowed}}
  }
  try {
  } catch (__externref_t &) { // expected-error {{reference to WebAssembly reference type is not allowed}}
  }
  try {
  } catch (__externref_t[0]) { // expected-error {{array has sizeless element type '__externref_t'}}
  }

  new __externref_t;    // expected-error {{allocation of sizeless type '__externref_t'}}
  new __externref_t[0]; // expected-error {{allocation of sizeless type '__externref_t'}}

  delete ref;     // expected-error {{cannot delete expression of type '__externref_t'}}
}
