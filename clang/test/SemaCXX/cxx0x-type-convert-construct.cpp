// RUN: %clang_cc1 -std=gnu++11 -fsyntax-only -verify %s 

void f() {
  char *u8str;
  u8str = u8"a UTF-8 string"; // expected-error {{assigning to 'char *' from incompatible type 'const char [15]'}}
  char16_t *ustr;
  ustr = u"a UTF-16 string"; // expected-error {{assigning to 'char16_t *' from incompatible type 'const char16_t [16]'}}
  char32_t *Ustr;
  Ustr = U"a UTF-32 string"; // expected-error {{assigning to 'char32_t *' from incompatible type 'const char32_t [16]'}}

  char *Rstr;
  Rstr = R"foo(a raw string)foo"; // expected-warning{{conversion from string literal to 'char *' is deprecated}}
  wchar_t *LRstr;
  LRstr = LR"foo(a wide raw string)foo"; // expected-warning{{conversion from string literal to 'wchar_t *' is deprecated}}
  char *u8Rstr;
  u8Rstr = u8R"foo(a UTF-8 raw string)foo"; // expected-error {{assigning to 'char *' from incompatible type 'const char [19]'}}
  char16_t *uRstr;
  uRstr = uR"foo(a UTF-16 raw string)foo"; // expected-error {{assigning to 'char16_t *' from incompatible type 'const char16_t [20]'}}
  char32_t *URstr;
  URstr = UR"foo(a UTF-32 raw string)foo"; // expected-error {{assigning to 'char32_t *' from incompatible type 'const char32_t [20]'}}
}
