// RUN: %clang_cc1 -xc %s -verify -DBOOL=_Bool
// RUN: %clang_cc1 -xc++ %s -verify -DBOOL=bool
// RUN: %clang_cc1 -xobjective-c %s -verify -DBOOL=_Bool
// RUN: %clang_cc1 -xc %s -verify -DBOOL=_Bool -Wformat-type-confusion -DTYPE_CONF
// RUN: %clang_cc1 -xc++ %s -verify -DBOOL=bool -Wformat-type-confusion -DTYPE_CONF

__attribute__((format(__printf__, 1, 2)))
int p(const char *fmt, ...);

BOOL b;

#ifdef __OBJC__
@interface NSString
+(NSString *)stringWithFormat:(NSString *)fmt, ...
    __attribute__((format(__NSString__, 1, 2)));
@end

#define YES __objc_yes
#define NO __objc_no
#endif

int main(void) {
  p("%d", b);
  p("%hd", b);
#ifdef TYPE_CONF
  // expected-warning@-2 {{format specifies type 'short' but the argument has type}}
#endif
  p("%hhd", b);
  p("%u", b);
  p("%hu", b);
#ifdef TYPE_CONF
  // expected-warning@-2 {{format specifies type 'unsigned short' but the argument has type}}
#endif
  p("%hhu", b);
  p("%c", b); // expected-warning {{using '%c' format specifier, but argument has boolean value}}
  p("%lc", b); // expected-warning {{using '%lc' format specifier, but argument has boolean value}}
  p("%c", 1 == 1); // expected-warning {{using '%c' format specifier, but argument has boolean value}}
  p("%f", b); // expected-warning{{format specifies type 'double' but the argument has type}}
  p("%ld", b); // expected-warning{{format specifies type 'long' but the argument has type}}
  p("%lld", b); // expected-warning{{format specifies type 'long long' but the argument has type}}

#ifdef __OBJC__
  [NSString stringWithFormat: @"%c", 0]; // probably fine?
  [NSString stringWithFormat: @"%c", NO]; // expected-warning {{using '%c' format specifier, but argument has boolean value}}
#endif
}
