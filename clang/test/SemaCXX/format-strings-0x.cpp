// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -std=c++11 %s

extern "C" {
extern int scanf(const char *restrict, ...);
extern int printf(const char *restrict, ...);
}

void f(char **sp, float *fp) {
  scanf("%as", sp); // expected-warning{{format specifies type 'float *' but the argument has type 'char **'}}

  printf("%p", sp); // expected-warning{{format specifies type 'void *' but the argument has type 'char **'}}
  scanf("%p", sp);  // expected-warning{{format specifies type 'void **' but the argument has type 'char **'}}

  printf("%a", 1.0);
  scanf("%afoobar", fp);
  printf(nullptr);
  printf(*sp); // expected-warning {{not a string literal}}
  // expected-note@-1{{treat the string as an argument to avoid this}}

  // PR13099
  printf(
    R"foobar(%)foobar"
    R"bazquux(d)bazquux" // expected-warning {{more '%' conversions than data arguments}}
    R"xyzzy()xyzzy");

  printf(u8"this is %d test", 0); // ok
  printf(u8R"foo(
      \u1234\U0010fffe
      %d)foo" // expected-warning {{more '%' conversions than data arguments}}
  );

  printf("init list: %d", { 0 }); // expected-error {{cannot pass initializer list to variadic function; expected type from format string was 'int'}}
  printf("void: %d", f(sp, fp)); // expected-error {{cannot pass expression of type 'void' to variadic function; expected type from format string was 'int'}}
  printf(0, { 0 }); // expected-error {{cannot pass initializer list to variadic function}}
}
