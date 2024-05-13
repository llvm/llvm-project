// RUN: %clang_cc1 -verify -ffixed-point %s

void func(void) {
  _Bool b;
  char c;
  int i;
  float f;
  double d;
  double _Complex dc;
  int _Complex ic;
  struct S {
    int i;
  } s;
  enum E {
    A
  } e;
  int *ptr;
  typedef int int_t;
  int_t i2;

  _Accum accum;
  _Fract fract = accum; // ok
  _Accum *accum_ptr;

  accum = dc;      // expected-error{{conversion between fixed point and '_Complex double' is not yet supported}}
  accum = ic;      // expected-error{{conversion between fixed point and '_Complex int' is not yet supported}}
  accum = s;       // expected-error{{assigning to '_Accum' from incompatible type 'struct S'}}
  accum = ptr;     // expected-error{{assigning to '_Accum' from incompatible type 'int *'}}
  accum_ptr = ptr; // expected-warning{{incompatible pointer types assigning to '_Accum *' from 'int *'}}

  dc = accum;      // expected-error{{conversion between fixed point and '_Complex double' is not yet supported}}
  ic = accum;      // expected-error{{conversion between fixed point and '_Complex int' is not yet supported}}
  s = accum;       // expected-error{{assigning to 'struct S' from incompatible type '_Accum'}}
  ptr = accum;     // expected-error{{assigning to 'int *' from incompatible type '_Accum'}}
  ptr = accum_ptr; // expected-warning{{incompatible pointer types assigning to 'int *' from '_Accum *'}}
}
