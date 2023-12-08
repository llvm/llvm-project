// RUN: %clang_cc1 -std=c++11 -verify %s

using size_t = decltype(sizeof(int));
void operator "" wibble(const char *); // expected-warning {{user-defined literal suffixes not starting with '_' are reserved; no literal will invoke this operator}}
void operator "" wibble(const char *, size_t); // expected-warning {{user-defined literal suffixes not starting with '_' are reserved; no literal will invoke this operator}}

template<typename T>
void f() {
  // A program containing a reserved ud-suffix is ill-formed.
  123wibble; // expected-error {{invalid suffix 'wibble'}}
  123.0wibble; // expected-error {{invalid suffix 'wibble'}}
  const char *p = ""wibble; // expected-error {{invalid suffix on literal; C++11 requires a space between literal and identifier}} expected-error {{expected ';'}}
  const char *q = R"x("hello")x"wibble; // expected-error {{invalid suffix on literal; C++11 requires a space between literal and identifier}} expected-error {{expected ';'}}
}
