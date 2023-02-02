// RUN: %clang_cc1 -fsyntax-only -verify -triple wasm32 -target-feature +reference-types %s

// Note: As WebAssembly references are sizeless types, we don't exhaustively
// test for cases covered by sizeless-1.c and similar tests.

// Unlike standard sizeless types, reftype globals are supported.
__externref_t r1;
extern __externref_t r2;
static __externref_t r3;

__externref_t *t1;               // expected-error {{pointer to WebAssembly reference type is not allowed}}
__externref_t **t2;              // expected-error {{pointer to WebAssembly reference type is not allowed}}
__externref_t ******t3;          // expected-error {{pointer to WebAssembly reference type is not allowed}}
static __externref_t t4[3];      // expected-error {{array has sizeless element type '__externref_t'}}
static __externref_t t5[];       // expected-error {{array has sizeless element type '__externref_t'}}
static __externref_t t6[] = {0}; // expected-error {{array has sizeless element type '__externref_t'}}
__externref_t t7[0];             // expected-error {{array has sizeless element type '__externref_t'}}
static __externref_t t8[0][0];   // expected-error {{array has sizeless element type '__externref_t'}}

static __externref_t table[0]; // expected-error {{array has sizeless element type '__externref_t'}}

struct s {
  __externref_t f1;       // expected-error {{field has sizeless type '__externref_t'}}
  __externref_t f2[0];    // expected-error {{array has sizeless element type '__externref_t'}}
  __externref_t f3[];     // expected-error {{array has sizeless element type '__externref_t'}}
  __externref_t f4[0][0]; // expected-error {{array has sizeless element type '__externref_t'}}
  __externref_t *f5;      // expected-error {{pointer to WebAssembly reference type is not allowed}}
  __externref_t ****f6;   // expected-error {{pointer to WebAssembly reference type is not allowed}}
};

union u {
  __externref_t f1;       // expected-error {{field has sizeless type '__externref_t'}}
  __externref_t f2[0];    // expected-error {{array has sizeless element type '__externref_t'}}
  __externref_t f3[];     // expected-error {{array has sizeless element type '__externref_t'}}
  __externref_t f4[0][0]; // expected-error {{array has sizeless element type '__externref_t'}}
  __externref_t *f5;      // expected-error {{pointer to WebAssembly reference type is not allowed}}
  __externref_t ****f6;   // expected-error {{pointer to WebAssembly reference type is not allowed}}
};

void illegal_argument_1(__externref_t table[]);     // expected-error {{array has sizeless element type '__externref_t'}}
void illegal_argument_2(__externref_t table[0][0]); // expected-error {{array has sizeless element type '__externref_t'}}
void illegal_argument_3(__externref_t *table);      // expected-error {{pointer to WebAssembly reference type is not allowed}}
void illegal_argument_4(__externref_t ***table);    // expected-error {{pointer to WebAssembly reference type is not allowed}}

__externref_t *illegal_return_1();   // expected-error {{pointer to WebAssembly reference type is not allowed}}
__externref_t ***illegal_return_2(); // expected-error {{pointer to WebAssembly reference type is not allowed}}

void varargs(int, ...);

__externref_t func(__externref_t ref) {
  &ref; // expected-error {{cannot take address of WebAssembly reference}}
  int foo = 40;
  (__externref_t *)(&foo);     // expected-error {{pointer to WebAssembly reference type is not allowed}}
  (__externref_t ****)(&foo);  // expected-error {{pointer to WebAssembly reference type is not allowed}}
  sizeof(ref);                 // expected-error {{invalid application of 'sizeof' to sizeless type '__externref_t'}}
  sizeof(__externref_t);       // expected-error {{invalid application of 'sizeof' to sizeless type '__externref_t'}}
  sizeof(__externref_t[0]);    // expected-error {{array has sizeless element type '__externref_t'}}
  sizeof(__externref_t[0][0]); // expected-error {{array has sizeless element type '__externref_t'}}
  sizeof(__externref_t *);     // expected-error {{pointer to WebAssembly reference type is not allowed}}
  sizeof(__externref_t ***);   // expected-error {{pointer to WebAssembly reference type is not allowed}};
  // expected-warning@+1 {{'_Alignof' applied to an expression is a GNU extension}}
  _Alignof(ref);                 // expected-error {{invalid application of 'alignof' to sizeless type '__externref_t'}}
  _Alignof(__externref_t);       // expected-error {{invalid application of 'alignof' to sizeless type '__externref_t'}}
  _Alignof(__externref_t[]);     // expected-error {{array has sizeless element type '__externref_t'}}
  _Alignof(__externref_t[0][0]); // expected-error {{array has sizeless element type '__externref_t'}}
  _Alignof(__externref_t *);     // expected-error {{pointer to WebAssembly reference type is not allowed}}
  _Alignof(__externref_t ***);   // expected-error {{pointer to WebAssembly reference type is not allowed}};
  varargs(1, ref);               // expected-error {{cannot pass expression of type '__externref_t' to variadic function}}

  return ref;
}
