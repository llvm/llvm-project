// RUN: %clang_cc1 -std=c++1z %s -include %s -verify

#ifndef INCLUDED
#define INCLUDED

#pragma clang system_header
namespace std {
  using size_t = decltype(sizeof(0));

  struct string_view {};
  string_view operator""sv(const char*, size_t);
}

#else

using namespace std;
string_view s = "foo"sv;
const char* p = "bar"sv; // expected-error {{no viable conversion}}
char error = 'x'sv; // expected-error {{invalid suffix}} expected-error {{expected ';'}}

#endif
