// RUN: %clang_cc1 -verify -ffixed-point %s

void func() {
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

  accum = b;       // expected-error{{conversion between fixed point and '_Bool' is not yet supported}}
  accum = i;       // expected-error{{conversion between fixed point and 'int' is not yet supported}}
  accum = i;       // expected-error{{conversion between fixed point and 'int' is not yet supported}}
  accum = f;       // expected-error{{conversion between fixed point and 'float' is not yet supported}}
  accum = d;       // expected-error{{conversion between fixed point and 'double' is not yet supported}}
  accum = dc;      // expected-error{{conversion between fixed point and '_Complex double' is not yet supported}}
  accum = ic;      // expected-error{{conversion between fixed point and '_Complex int' is not yet supported}}
  accum = s;       // expected-error{{assigning to '_Accum' from incompatible type 'struct S'}}
  accum = e;       // expected-error{{conversion between fixed point and 'enum E' is not yet supported}}
  accum = ptr;     // expected-error{{assigning to '_Accum' from incompatible type 'int *'}}
  accum_ptr = ptr; // expected-warning{{incompatible pointer types assigning to '_Accum *' from 'int *'}}
  accum = i2;      // expected-error{{conversion between fixed point and 'int_t' (aka 'int') is not yet supported}}

  c = accum;       // expected-error{{conversion between fixed point and 'char' is not yet supported}}
  i = accum;       // expected-error{{conversion between fixed point and 'int' is not yet supported}}
  f = accum;       // expected-error{{conversion between fixed point and 'float' is not yet supported}}
  d = accum;       // expected-error{{conversion between fixed point and 'double' is not yet supported}}
  dc = accum;      // expected-error{{conversion between fixed point and '_Complex double' is not yet supported}}
  ic = accum;      // expected-error{{conversion between fixed point and '_Complex int' is not yet supported}}
  s = accum;       // expected-error{{assigning to 'struct S' from incompatible type '_Accum'}}
  e = accum;       // expected-error{{conversion between fixed point and 'enum E' is not yet supported}}
  ptr = accum;     // expected-error{{assigning to 'int *' from incompatible type '_Accum'}}
  ptr = accum_ptr; // expected-warning{{incompatible pointer types assigning to 'int *' from '_Accum *'}}
  i2 = accum;      // expected-error{{conversion between fixed point and 'int' is not yet supported}}
}
