// RUN: %clang_cc1 -fsyntax-only -verify -pedantic -std=c++11 %s

extern "C" {
extern int scanf(const char *restrict, ...);
extern int printf(const char *restrict, ...);
}

void f(char **sp, float *fp) {
  scanf("%as", sp); // expected-warning{{format specifies type 'float *' but the argument has type 'char **'}}

  printf("%a", 1.0);
  scanf("%afoobar", fp);
  printf(nullptr);
  printf(*sp); // expected-warning {{not a string literal}}

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
}
