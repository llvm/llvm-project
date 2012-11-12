// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

void f() {

  const char* a = u8"abc" u"abc"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char* b = u8"abc" U"abc"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char* c = u8"abc" L"abc"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char* d = u8"abc" uR"(abc)"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char* e = u8"abc" UR"(abc)"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char* f = u8"abc" LR"(abc)"; // expected-error {{unsupported non-standard concatenation of string literals}}

  const char16_t* g = u"abc" u8"abc"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char16_t* h = u"abc" U"abc"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char16_t* i = u"abc" L"abc"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char16_t* j = u"abc" u8R"(abc)"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char16_t* k = u"abc" UR"(abc)"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char16_t* l = u"abc" LR"(abc)"; // expected-error {{unsupported non-standard concatenation of string literals}}

  const char32_t* m = U"abc" u8"abc"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char32_t* n = U"abc" u"abc"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char32_t* o = U"abc" L"abc"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char32_t* p = U"abc" u8R"(abc)"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char32_t* q = U"abc" uR"(abc)"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const char32_t* r = U"abc" LR"(abc)"; // expected-error {{unsupported non-standard concatenation of string literals}}

  const wchar_t* s = L"abc" u8"abc"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const wchar_t* t = L"abc" u"abc"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const wchar_t* u = L"abc" U"abc"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const wchar_t* v = L"abc" u8R"(abc)"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const wchar_t* w = L"abc" uR"(abc)"; // expected-error {{unsupported non-standard concatenation of string literals}}
  const wchar_t* x = L"abc" UR"(abc)"; // expected-error {{unsupported non-standard concatenation of string literals}}
}

