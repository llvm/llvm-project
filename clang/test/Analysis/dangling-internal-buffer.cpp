//RUN: %clang_analyze_cc1 -analyzer-checker=alpha.cplusplus.DanglingInternalBuffer %s -analyzer-output=text -verify

namespace std {

template< typename CharT >
class basic_string {
public:
  ~basic_string();
  const CharT *c_str() const;
};

typedef basic_string<char> string;
typedef basic_string<wchar_t> wstring;
typedef basic_string<char16_t> u16string;
typedef basic_string<char32_t> u32string;

} // end namespace std

void consume(const char *) {}
void consume(const wchar_t *) {}
void consume(const char16_t *) {}
void consume(const char32_t *) {}

void deref_after_scope_char() {
  const char *c;
  {
    std::string s;
    c = s.c_str(); // expected-note {{Pointer to dangling buffer was obtained here}}
  } // expected-note {{Internal buffer is released because the object was destroyed}}
  consume(c); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_scope_char2() {
  const char *c;
  {
    std::string s;
    c = s.c_str(); // expected-note {{Pointer to dangling buffer was obtained here}}
  } // expected-note {{Internal buffer is released because the object was destroyed}}
  std::string s;
  const char *c2 = s.c_str();
  consume(c); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_scope_wchar_t() {
  const wchar_t *w;
  {
    std::wstring ws;
    w = ws.c_str(); // expected-note {{Pointer to dangling buffer was obtained here}}
  } // expected-note {{Internal buffer is released because the object was destroyed}}
  consume(w); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_scope_char16_t() {
  const char16_t *c16;
  {
    std::u16string s16;
    c16 = s16.c_str(); // expected-note {{Pointer to dangling buffer was obtained here}}
  } // expected-note {{Internal buffer is released because the object was destroyed}}
  consume(c16); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_scope_char32_t() {
  const char32_t *c32;
  {
    std::u32string s32;
    c32 = s32.c_str(); // expected-note {{Pointer to dangling buffer was obtained here}}
  } // expected-note {{Internal buffer is released because the object was destroyed}}
  consume(c32); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_scope_ok() {
  const char *c;
  std::string s;
  {
    c = s.c_str();
  }
  consume(c); // no-warning
}
