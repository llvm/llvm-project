// RUN: %clang_cc1 %s -fsyntax-only -verify

#define _AS1 __attribute__((address_space(1)))
#define _AS2 __attribute__((address_space(2)))
#define _AS3 __attribute__((address_space(3)))

void bar(_AS2 int a); // expected-error {{parameter may not be qualified with an address space}}

void foo(_AS3 float *a,
         _AS1 float b) // expected-error {{parameter may not be qualified with an address space}}
{
  _AS2 *x;// expected-warning {{type specifier missing, defaults to 'int'}}
  _AS1 float * _AS2 *B;

  int _AS1 _AS2 *Y;   // expected-error {{multiple address spaces specified for type}}
  int *_AS1 _AS2 *Z;  // expected-error {{multiple address spaces specified for type}}
  int *_AS1 _AS1 *M;  // expected-warning {{multiple identical address spaces specified for type}}

  _AS1 int local;     // expected-error {{automatic variable qualified with an address space}}
  _AS1 int array[5];  // expected-error {{automatic variable qualified with an address space}}
  _AS1 int arrarr[5][5]; // expected-error {{automatic variable qualified with an address space}}

  __attribute__((address_space(-1))) int *_boundsA; // expected-error {{address space is negative}}
  __attribute__((address_space(0x7FFFFF))) int *_boundsB; // expected-error {{address space is larger than the maximum supported}}
  __attribute__((address_space(0x1000000))) int *_boundsC; // expected-error {{address space is larger than the maximum supported}}
  // chosen specifically to overflow 32 bits and come out reasonable
  __attribute__((address_space(4294967500))) int *_boundsD; // expected-error {{address space is larger than the maximum supported}}

  *a = 5.0f + b;
}

struct _st {
 int x, y;
} s __attribute ((address_space(1))) = {1, 1};


// rdar://6774906
__attribute__((address_space(256))) void * * const base = 0;
void * get_0(void) {
  return base[0];  // expected-error {{returning '__attribute__((address_space(256))) void *' from a function with result type 'void *' changes address space of pointer}}
}

__attribute__((address_space(1))) char test3_array[10];
void test3(void) {
  extern void test3_helper(char *p); // expected-note {{passing argument to parameter 'p' here}}
  test3_helper(test3_array); // expected-error {{changes address space of pointer}}
}

typedef void ft(void);
_AS1 ft qf; // expected-error {{function type may not be qualified with an address space}}
typedef _AS1 ft qft; // expected-error {{function type may not be qualified with an address space}}


typedef _AS2 int AS2Int;

struct HasASFields
{
  _AS2 int as_field; // expected-error {{field may not be qualified with an address space}}
   AS2Int typedef_as_field; // expected-error {{field may not be qualified with an address space}}
};

// Assertion failure was when the field was accessed
void access_as_field()
{
    struct HasASFields x;
    (void) bar.as_field;
}

typedef int PR4997 __attribute__((address_space(Foobar))); // expected-error {{use of undeclared identifier 'Foobar'}}
__attribute__((address_space("12"))) int *i; // expected-error {{'address_space' attribute requires an integer constant}}

// Clang extension doesn't forbid operations on pointers to different address spaces.
char* cmp(_AS1 char *x,  _AS2 char *y) {
  return x < y ? x : y; // expected-error{{conditional operator with the second and third operands of type  ('__attribute__((address_space(1))) char *' and '__attribute__((address_space(2))) char *') which are pointers to non-overlapping address spaces}}
}

struct SomeStruct {
  int a;
  long b;
  int c;
};

// Compound literals in function scope are lvalues with automatic storage duration,
// so they cannot realistically be qualified with an address space.
void as_compound_literal() {
  (_AS1 struct SomeStruct){1, 2, 3}; // expected-error {{compound literal in function scope may not be qualified with an address space}}
  (_AS1 char[]){"test"}; // expected-error {{compound literal in function scope may not be qualified with an address space}}
  (_AS1 char[]){'a', 'b', 'c'}; // expected-error {{compound literal in function scope may not be qualified with an address space}}
}
