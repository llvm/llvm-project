// RUN: %clang_cc1 -Wunsafe-buffer-usage -std=c++23 %s -verify

// expected-no-diagnostics

template <class _CharT> struct basic_string {
  struct __rep {
    __rep(int);
  };
  basic_string(_CharT, __rep = int()) {}
};
char16_t operators___str;
decltype(sizeof(int)) operators___len;
void operators() { basic_string(operators___str, operators___len); }
