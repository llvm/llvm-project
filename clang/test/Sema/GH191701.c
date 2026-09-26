// RUN: %clang_cc1 -fsyntax-only -verify -ffixed-point -fexperimental-overflow-behavior-types %s

// Fixed point types have no conversions with _BitInt or overflow behavior types.

#define test(e, t) _Generic (e, default : 0, t : 1)

void GH191701(_Fract ci) {
  _BitInt(31) bi = 0;
  _Static_assert(test(ci + bi, _Complex int), ""); // expected-error {{invalid operands to binary expression ('_Fract' and '_BitInt(31)')}}
}

void bitint(int c, _Fract f, _Accum a, _Sat short _Fract sf, _BitInt(31) bi,
            unsigned _BitInt(8) ubi) {
  (void)(f + bi);   // expected-error {{invalid operands to binary expression ('_Fract' and '_BitInt(31)')}}
  (void)(bi + f);   // expected-error {{invalid operands to binary expression ('_BitInt(31)' and '_Fract')}}
  (void)(a - bi);   // expected-error {{invalid operands to binary expression ('_Accum' and '_BitInt(31)')}}
  (void)(bi * a);   // expected-error {{invalid operands to binary expression ('_BitInt(31)' and '_Accum')}}
  (void)(sf / ubi); // expected-error {{invalid operands to binary expression ('_Sat short _Fract' and 'unsigned _BitInt(8)')}}
  (void)(f < bi);   // expected-error {{invalid operands to binary expression ('_Fract' and '_BitInt(31)')}}
  (void)(ubi == a); // expected-error {{invalid operands to binary expression ('unsigned _BitInt(8)' and '_Accum')}}
  f += bi;          // expected-error {{invalid operands to binary expression ('_Fract' and '_BitInt(31)')}}
  bi -= a;          // expected-error {{invalid operands to binary expression ('_BitInt(31)' and '_Accum')}}
  (void)(c ? f : bi); // expected-error {{incompatible operand types ('_Fract' and '_BitInt(31)')}}
}

void overflow_behavior(int c, _Fract f, _Accum a, int __ob_wrap w,
                       long __ob_trap t) {
  (void)(f + w);   // expected-error {{invalid operands to binary expression ('_Fract' and '__ob_wrap int')}}
  (void)(t * a);   // expected-error {{invalid operands to binary expression ('__ob_trap long' and '_Accum')}}
  a -= w;          // expected-error {{invalid operands to binary expression ('_Accum' and '__ob_wrap int')}}
  (void)(c ? w : f); // expected-error {{incompatible operand types ('__ob_wrap int' and '_Fract')}}
}
