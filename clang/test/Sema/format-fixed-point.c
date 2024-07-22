// RUN: %clang_cc1 -ffixed-point -fsyntax-only -verify -Wformat -isystem %S/Inputs %s
// RUN: %clang_cc1 -fsyntax-only -verify -Wformat -isystem %S/Inputs %s -DWITHOUT_FIXED_POINT

int printf(const char *restrict, ...);

short s;
unsigned short us;
int i;
unsigned int ui;
long l;
unsigned long ul;
float fl;
double d;
char c;
unsigned char uc;

#ifndef WITHOUT_FIXED_POINT
short _Fract sf;
_Fract f;
long _Fract lf;
unsigned short _Fract usf;
unsigned _Fract uf;
unsigned long _Fract ulf;
short _Accum sa;
_Accum a;
long _Accum la;
unsigned short _Accum usa;
unsigned _Accum ua;
unsigned long _Accum ula;
_Sat short _Fract sat_sf;
_Sat _Fract sat_f;
_Sat long _Fract sat_lf;
_Sat unsigned short _Fract sat_usf;
_Sat unsigned _Fract sat_uf;
_Sat unsigned long _Fract sat_ulf;
_Sat short _Accum sat_sa;
_Sat _Accum sat_a;
_Sat long _Accum sat_la;
_Sat unsigned short _Accum sat_usa;
_Sat unsigned _Accum sat_ua;
_Sat unsigned long _Accum sat_ula;

void test_invalid_args(void) {
  /// None of these should match against a fixed point type.
  printf("%r", s);   // expected-warning{{format specifies type '_Fract' but the argument has type 'short'}}
  printf("%r", us);  // expected-warning{{format specifies type '_Fract' but the argument has type 'unsigned short'}}
  printf("%r", i);   // expected-warning{{format specifies type '_Fract' but the argument has type 'int'}}
  printf("%r", ui);  // expected-warning{{format specifies type '_Fract' but the argument has type 'unsigned int'}}
  printf("%r", l);   // expected-warning{{format specifies type '_Fract' but the argument has type 'long'}}
  printf("%r", ul);  // expected-warning{{format specifies type '_Fract' but the argument has type 'unsigned long'}}
  printf("%r", fl);  // expected-warning{{format specifies type '_Fract' but the argument has type 'float'}}
  printf("%r", d);   // expected-warning{{format specifies type '_Fract' but the argument has type 'double'}}
  printf("%r", c);   // expected-warning{{format specifies type '_Fract' but the argument has type 'char'}}
  printf("%r", uc);  // expected-warning{{format specifies type '_Fract' but the argument has type 'unsigned char'}}
}

void test_fixed_point_specifiers(void) {
  printf("%r", f);
  printf("%R", uf);
  printf("%k", a);
  printf("%K", ua);

  /// Test different sizes.
  printf("%r", sf);   // expected-warning{{format specifies type '_Fract' but the argument has type 'short _Fract'}}
  printf("%r", lf);   // expected-warning{{format specifies type '_Fract' but the argument has type 'long _Fract'}}
  printf("%R", usf);  // expected-warning{{format specifies type 'unsigned _Fract' but the argument has type 'unsigned short _Fract'}}
  printf("%R", ulf);  // expected-warning{{format specifies type 'unsigned _Fract' but the argument has type 'unsigned long _Fract'}}
  printf("%k", sa);   // expected-warning{{format specifies type '_Accum' but the argument has type 'short _Accum'}}
  printf("%k", la);   // expected-warning{{format specifies type '_Accum' but the argument has type 'long _Accum'}}
  printf("%K", usa);  // expected-warning{{format specifies type 'unsigned _Accum' but the argument has type 'unsigned short _Accum'}}
  printf("%K", ula);  // expected-warning{{format specifies type 'unsigned _Accum' but the argument has type 'unsigned long _Accum'}}

  /// Test signs.
  printf("%r", uf);  // expected-warning{{format specifies type '_Fract' but the argument has type 'unsigned _Fract'}}
  printf("%R", f);   // expected-warning{{format specifies type 'unsigned _Fract' but the argument has type '_Fract'}}
  printf("%k", ua);  // expected-warning{{format specifies type '_Accum' but the argument has type 'unsigned _Accum'}}
  printf("%K", a);   // expected-warning{{format specifies type 'unsigned _Accum' but the argument has type '_Accum'}}

  /// Test between types.
  printf("%r", a);   // expected-warning{{format specifies type '_Fract' but the argument has type '_Accum'}}
  printf("%R", ua);  // expected-warning{{format specifies type 'unsigned _Fract' but the argument has type 'unsigned _Accum'}}
  printf("%k", f);   // expected-warning{{format specifies type '_Accum' but the argument has type '_Fract'}}
  printf("%K", uf);  // expected-warning{{format specifies type 'unsigned _Accum' but the argument has type 'unsigned _Fract'}}

  /// Test saturated types.
  printf("%r", sat_f);
  printf("%R", sat_uf);
  printf("%k", sat_a);
  printf("%K", sat_ua);
}

void test_length_modifiers_and_flags(void) {
  printf("%hr", sf);
  printf("%lr", lf);
  printf("%hR", usf);
  printf("%lR", ulf);
  printf("%hk", sa);
  printf("%lk", la);
  printf("%hK", usa);
  printf("%lK", ula);

  printf("%hr", sat_sf);
  printf("%lr", sat_lf);
  printf("%hR", sat_usf);
  printf("%lR", sat_ulf);
  printf("%hk", sat_sa);
  printf("%lk", sat_la);
  printf("%hK", sat_usa);
  printf("%lK", sat_ula);

  printf("%10r", f);
  printf("%10.10r", f);
  printf("%010r", f);
  printf("%-10r", f);
  printf("%.10r", f);
  printf("%+r", f);
  printf("% r", f);
  printf("%#r", f);
  printf("%#.r", f);
  printf("%#.0r", f);

  /// Test some invalid length modifiers.
  printf("%zr", f);   // expected-warning{{length modifier 'z' results in undefined behavior or no effect with 'r' conversion specifier}}
  printf("%llr", f);  // expected-warning{{length modifier 'll' results in undefined behavior or no effect with 'r' conversion specifier}}
  printf("%hhr", f);  // expected-warning{{length modifier 'hh' results in undefined behavior or no effect with 'r' conversion specifier}}

  // + on an unsigned fixed point type.
  printf("%+hR", usf);  // expected-warning{{flag '+' results in undefined behavior with 'R' conversion specifier}}
  printf("%+R", uf);    // expected-warning{{flag '+' results in undefined behavior with 'R' conversion specifier}}
  printf("%+lR", ulf);  // expected-warning{{flag '+' results in undefined behavior with 'R' conversion specifier}}
  printf("%+hK", usa);  // expected-warning{{flag '+' results in undefined behavior with 'K' conversion specifier}}
  printf("%+K", ua);    // expected-warning{{flag '+' results in undefined behavior with 'K' conversion specifier}}
  printf("%+lK", ula);  // expected-warning{{flag '+' results in undefined behavior with 'K' conversion specifier}}
  printf("% hR", usf);  // expected-warning{{flag ' ' results in undefined behavior with 'R' conversion specifier}}
  printf("% R", uf);    // expected-warning{{flag ' ' results in undefined behavior with 'R' conversion specifier}}
  printf("% lR", ulf);  // expected-warning{{flag ' ' results in undefined behavior with 'R' conversion specifier}}
  printf("% hK", usa);  // expected-warning{{flag ' ' results in undefined behavior with 'K' conversion specifier}}
  printf("% K", ua);    // expected-warning{{flag ' ' results in undefined behavior with 'K' conversion specifier}}
  printf("% lK", ula);  // expected-warning{{flag ' ' results in undefined behavior with 'K' conversion specifier}}
}
#else
void test_fixed_point_specifiers_no_printf() {
  printf("%k", i);  // expected-warning{{invalid conversion specifier 'k'}}  
  printf("%K", i);  // expected-warning{{invalid conversion specifier 'K'}}
  printf("%r", i);  // expected-warning{{invalid conversion specifier 'r'}}
  printf("%R", i);  // expected-warning{{invalid conversion specifier 'R'}}
}
#endif  // WITHOUT_FIXED_POINT
