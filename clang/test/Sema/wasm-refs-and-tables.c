// RUN: %clang_cc1 -fsyntax-only -verify=expected,conly -triple wasm32 -Wno-unused-value -target-feature +reference-types %s
// RUN: %clang_cc1 -x c++ -std=c++17 -fsyntax-only -verify=expected,cpp -triple wasm32 -Wno-unused-value -target-feature +reference-types %s

// Note: As WebAssembly references are sizeless types, we don't exhaustively
// test for cases covered by sizeless-1.c and similar tests.

// Unlike standard sizeless types, reftype globals are supported.
__externref_t r1;
extern __externref_t r2;
static __externref_t r3;

__externref_t *t1;
__externref_t **t2;
__externref_t ******t3;
static __externref_t t4[3];      // expected-error {{only zero-length WebAssembly tables are currently supported}}
static __externref_t t5[];       // expected-error {{only zero-length WebAssembly tables are currently supported}}
static __externref_t t6[] = {0}; // expected-error {{only zero-length WebAssembly tables are currently supported}}
__externref_t t7[0];             // expected-error {{WebAssembly table must be static}}
static __externref_t t8[0][0];   // expected-error {{multi-dimensional arrays of WebAssembly references are not allowed}}
static __externref_t (*t9)[0];   // expected-error {{cannot form a pointer to a WebAssembly table}}

static __externref_t table[0];
static __externref_t other_table[0] = {};
static __externref_t another_table[] = {}; // expected-error {{only zero-length WebAssembly tables are currently supported}}

struct s {
  __externref_t f1;       // expected-error {{field has sizeless type '__externref_t'}}
  __externref_t f2[0];    // expected-error {{field has sizeless type '__externref_t'}}
  __externref_t f3[];     // expected-error {{field has sizeless type '__externref_t'}}
  __externref_t f4[0][0]; // expected-error {{multi-dimensional arrays of WebAssembly references are not allowed}}
  __externref_t *f5;
  __externref_t ****f6;
  __externref_t (*f7)[0]; // expected-error {{cannot form a pointer to a WebAssembly table}}
};

union u {
  __externref_t f1;       // expected-error {{field has sizeless type '__externref_t'}}
  __externref_t f2[0];    // expected-error {{field has sizeless type '__externref_t'}}
  __externref_t f3[];     // expected-error {{field has sizeless type '__externref_t'}}
  __externref_t f4[0][0]; // expected-error {{multi-dimensional arrays of WebAssembly references are not allowed}}
  __externref_t *f5;
  __externref_t ****f6;
  __externref_t (*f7)[0]; // expected-error {{cannot form a pointer to a WebAssembly table}}
};

void illegal_argument_1(__externref_t table[0][0]); // expected-error {{multi-dimensional arrays of WebAssembly references are not allowed}}
void illegal_argument_2(__externref_t (*table)[0]); // expected-error {{cannot form a pointer to a WebAssembly table}}

void okay_argument_1(__externref_t *table);
void okay_argument_2(__externref_t ***table);
void okay_argument_3(__externref_t table[0]);

__externref_t *okay_return_1();
__externref_t ***okay_return_2();
__externref_t (*illegal_return3())[0]; // expected-error {{cannot form a pointer to a WebAssembly table}}

void varargs(int, ...);
typedef void (*__funcref funcref_t)();
typedef void (*__funcref __funcref funcref_fail_t)(); // expected-warning {{attribute '__funcref' is already applied}}

__externref_t func(__externref_t ref) {
  &ref;                        // expected-error {{cannot take address of WebAssembly reference}}
  int foo = 40;
  (__externref_t ****)(&foo);
  sizeof(ref);                 // expected-error {{invalid application of 'sizeof' to sizeless type '__externref_t'}}
  sizeof(__externref_t);       // expected-error {{invalid application of 'sizeof' to sizeless type '__externref_t'}}
  sizeof(__externref_t[0]);    // expected-error {{invalid application of 'sizeof' to WebAssembly table}}
  sizeof(table);               // expected-error {{invalid application of 'sizeof' to WebAssembly table}}
  sizeof(__externref_t[0][0]); // expected-error {{multi-dimensional arrays of WebAssembly references are not allowed}}
  sizeof(__externref_t *);
  sizeof(__externref_t ***);
  // expected-warning@+1 {{'_Alignof' applied to an expression is a GNU extension}}
  _Alignof(ref);                 // expected-error {{invalid application of 'alignof' to sizeless type '__externref_t'}}
  _Alignof(__externref_t);       // expected-error {{invalid application of 'alignof' to sizeless type '__externref_t'}}
  _Alignof(__externref_t[]);     // expected-error {{invalid application of 'alignof' to sizeless type '__externref_t'}}
  _Alignof(__externref_t[0]);    // expected-error {{invalid application of 'alignof' to sizeless type '__externref_t'}}
  _Alignof(table);               // expected-warning {{'_Alignof' applied to an expression is a GNU extension}} expected-error {{invalid application of 'alignof' to WebAssembly table}}
  _Alignof(__externref_t[0][0]); // expected-error {{multi-dimensional arrays of WebAssembly references are not allowed}}
  _Alignof(__externref_t *);
  _Alignof(__externref_t ***);
  varargs(1, ref);               // expected-error {{cannot pass expression of type '__externref_t' to variadic function}}

  __externref_t lt1[0];           // expected-error {{WebAssembly table cannot be declared within a function}}
  static __externref_t lt2[0];    // expected-error {{WebAssembly table cannot be declared within a function}}
  static __externref_t lt3[0][0]; // expected-error {{multi-dimensional arrays of WebAssembly references are not allowed}}
  static __externref_t(*lt4)[0];  // expected-error {{cannot form a pointer to a WebAssembly table}}
  // conly-error@+2 {{cannot use WebAssembly table as a function parameter}}
  // cpp-error@+1 {{no matching function for call to 'okay_argument_3'}}
  okay_argument_3(table);
  varargs(1, table);              // expected-error {{cannot use WebAssembly table as a function parameter}}
  table == 1;                     // expected-error {{cannot cast from a WebAssembly table}}
  1 >= table;                     // expected-error {{cannot cast from a WebAssembly table}}
  table == other_table;           // expected-error {{cannot cast from a WebAssembly table}}
  table !=- table;                // expected-error {{cannot cast from a WebAssembly table}}
  !table;                         // expected-error {{cannot cast from a WebAssembly table}}
  1 && table;                     // expected-error {{invalid operands to binary expression ('int' and '__attribute__((address_space(1))) __externref_t[0]')}}
  table || 1;                     // expected-error {{invalid operands to binary expression ('__attribute__((address_space(1))) __externref_t[0]' and 'int')}}
  table ? : other_table;          // expected-error {{cannot cast from a WebAssembly table}}
  (void *)table;                  // expected-error {{cannot cast from a WebAssembly table}}
  void *u;
  u = table;                      // expected-error {{cannot assign a WebAssembly table}}
  void *v = table;                // expected-error {{cannot assign a WebAssembly table}}
  &table;                         // expected-error {{cannot form a reference to a WebAssembly table}}
  (void)table;                    // conly-error {{cannot cast from a WebAssembly table}}

  table[0];                       // expected-error {{cannot subscript a WebAssembly table}}
  table[0] = ref;                 // expected-error {{cannot subscript a WebAssembly table}}

  int i = 0;                      // cpp-note {{declared here}}
  __externref_t oh_no_vlas[i];    // expected-error {{WebAssembly table cannot be declared within a function}} \
                                     cpp-warning {{variable length arrays in C++ are a Clang extension}} \
                                     cpp-note {{read of non-const variable 'i' is not allowed in a constant expression}}

  return ref;
}

void foo() {
  static __externref_t t[0];      // expected-error {{WebAssembly table cannot be declared within a function}}
  {
    static __externref_t t2[0];   // expected-error {{WebAssembly table cannot be declared within a function}}
    for (;;) {
      static __externref_t t3[0]; // expected-error {{WebAssembly table cannot be declared within a function}}
    }
  }
  int i = ({
    static __externref_t t4[0];   // expected-error {{WebAssembly table cannot be declared within a function}}
    1;
  });
}

void *ret_void_ptr() {
  return table; // expected-error {{cannot return a WebAssembly table}}
}
