// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -verify \
// RUN:   -target-feature +float128 -mabi=ieeelongdouble -fsyntax-only -Wno-unused %s
// RUN: %clang_cc1 -triple powerpc64le-unknown-unknown -verify \
// RUN:   -target-feature +float128 -fsyntax-only -Wno-unused %s

__float128 cast1(__ibm128 x) { return x; } // expected-error {{returning '__ibm128' from a function with incompatible result type '__float128'}}

__ibm128 cast2(__float128 x) { return x; } // expected-error {{returning '__float128' from a function with incompatible result type '__ibm128'}}

__ibm128 gf;

void narrow(double *d, float *f) {
  __ibm128 v = gf;
  gf = *d;      // expected-no-error {{assigning to '__ibm128' from incompatible type 'double'}}
  *f = v;       // expected-no-error {{assigning to 'float' from incompatible type '__ibm128'}}
  *d = gf + *f; // expected-no-error {{invalid operands to binary expression ('__ibm128' and 'float')}}
}

#ifdef __LONG_DOUBLE_IEEE128__
long double cast3(__ibm128 x) { return x; } // expected-error {{returning '__ibm128' from a function with incompatible result type 'long double'}}

__ibm128 cast4(long double x) { return x; } // expected-error {{returning 'long double' from a function with incompatible result type '__ibm128'}}

void imp_cast(__ibm128 w, __float128 q, long double l, _Bool b) {
  w + q;      // expected-error {{invalid operands to binary expression ('__ibm128' and '__float128')}}
  l + w;      // expected-error {{invalid operands to binary expression ('long double' and '__ibm128')}}
  q - w;      // expected-error {{invalid operands to binary expression ('__float128' and '__ibm128')}}
  w - l;      // expected-error {{invalid operands to binary expression ('__ibm128' and 'long double')}}
  w *l;       // expected-error {{invalid operands to binary expression ('__ibm128' and 'long double')}}
  q *w;       // expected-error {{invalid operands to binary expression ('__float128' and '__ibm128')}}
  q / w;      // expected-error {{invalid operands to binary expression ('__float128' and '__ibm128')}}
  w / l;      // expected-error {{invalid operands to binary expression ('__ibm128' and 'long double')}}
  w = q;      // expected-error {{assigning to '__ibm128' from incompatible type '__float128'}}
  q = w;      // expected-error {{assigning to '__float128' from incompatible type '__ibm128'}}
  l = w;      // expected-error {{assigning to 'long double' from incompatible type '__ibm128'}}
  w = l;      // expected-error {{assigning to '__ibm128' from incompatible type 'long double'}}
  b ? q : w;  // expected-error {{incompatible operand types ('__float128' and '__ibm128')}}
  !b ? w : l; // expected-error {{incompatible operand types ('__ibm128' and 'long double')}}
}
#elif __LONG_DOUBLE_IBM128__
long double cast3(__ibm128 x) { return x; } // expected-no-error {{returning '__ibm128' from a function with incompatible result type 'long double'}}

__ibm128 cast4(long double x) { return x; } // expected-no-error {{returning 'long double' from a function with incompatible result type '__ibm128'}}

void imp_cast(__ibm128 w, __float128 q, long double l, _Bool b) {
  w + q;      // expected-error {{invalid operands to binary expression ('__ibm128' and '__float128')}}
  l + w;      // expected-no-error {{invalid operands to binary expression ('long double' and '__ibm128')}}
  q - w;      // expected-error {{invalid operands to binary expression ('__float128' and '__ibm128')}}
  w - l;      // expected-no-error {{invalid operands to binary expression ('__ibm128' and 'long double')}}
  w *l;       // expected-no-error {{invalid operands to binary expression ('__ibm128' and 'long double')}}
  q *w;       // expected-error {{invalid operands to binary expression ('__float128' and '__ibm128')}}
  q / w;      // expected-error {{invalid operands to binary expression ('__float128' and '__ibm128')}}
  w / l;      // expected-no-error {{invalid operands to binary expression ('__ibm128' and 'long double')}}
  w = q;      // expected-error {{assigning to '__ibm128' from incompatible type '__float128'}}
  q = w;      // expected-error {{assigning to '__float128' from incompatible type '__ibm128'}}
  l = w;      // expected-no-error {{assigning to 'long double' from incompatible type '__ibm128'}}
  w = l;      // expected-no-error {{assigning to '__ibm128' from incompatible type 'long double'}}
  b ? q : w;  // expected-error {{incompatible operand types ('__float128' and '__ibm128')}}
  !b ? w : l; // expected-no-error {{incompatible operand types ('__ibm128' and 'long double')}}
}
#endif
