//RUN: %clang_analyze_cc1 -analyzer-checker=alpha.cplusplus.DanglingInternalBuffer %s -analyzer-output=text -verify

namespace std {

template< typename CharT >
class basic_string {
public:
  ~basic_string();
  const CharT *c_str() const;
  const CharT *data() const;
  CharT *data();
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

void deref_after_scope_char_cstr() {
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

void deref_after_scope_char_data() {
  const char *c;
  {
    std::string s;
    c = s.data(); // expected-note {{Pointer to dangling buffer was obtained here}}
  } // expected-note {{Internal buffer is released because the object was destroyed}}
  std::string s;
  const char *c2 = s.data();
  consume(c); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_scope_char_data_non_const() {
  char *c;
  {
    std::string s;
    c = s.data(); // expected-note {{Pointer to dangling buffer was obtained here}}
  } // expected-note {{Internal buffer is released because the object was destroyed}}
  std::string s;
  char *c2 = s.data();
  consume(c); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}


void deref_after_scope_wchar_t_cstr() {
  const wchar_t *w;
  {
    std::wstring ws;
    w = ws.c_str(); // expected-note {{Pointer to dangling buffer was obtained here}}
  } // expected-note {{Internal buffer is released because the object was destroyed}}
  std::wstring ws;
  const wchar_t *w2 = ws.c_str();
  consume(w); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_scope_wchar_t_data() {
  const wchar_t *w;
  {
    std::wstring ws;
    w = ws.data(); // expected-note {{Pointer to dangling buffer was obtained here}}
  } // expected-note {{Internal buffer is released because the object was destroyed}}
  std::wstring ws;
  const wchar_t *w2 = ws.data();
  consume(w); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_scope_char16_t_cstr() {
  const char16_t *c16;
  {
    std::u16string s16;
    c16 = s16.c_str(); // expected-note {{Pointer to dangling buffer was obtained here}}
  } // expected-note {{Internal buffer is released because the object was destroyed}}
  std::u16string s16;
  const char16_t *c16_2 = s16.c_str();
  consume(c16); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_scope_char32_t_data() {
  const char32_t *c32;
  {
    std::u32string s32;
    c32 = s32.data(); // expected-note {{Pointer to dangling buffer was obtained here}}
  } // expected-note {{Internal buffer is released because the object was destroyed}}
  std::u32string s32;
  const char32_t *c32_2 = s32.data();
  consume(c32); // expected-warning {{Use of memory after it is freed}}
  // expected-note@-1 {{Use of memory after it is freed}}
}

void multiple_symbols(bool cond) {
  const char *c1, *d1;
  {
    std::string s1;
    c1 = s1.c_str(); // expected-note {{Pointer to dangling buffer was obtained here}}
    d1 = s1.data(); // expected-note {{Pointer to dangling buffer was obtained here}}
    const char *local = s1.c_str();
    consume(local); // no-warning
  } // expected-note {{Internal buffer is released because the object was destroyed}}
  // expected-note@-1 {{Internal buffer is released because the object was destroyed}}
  std::string s2;
  const char *c2 = s2.c_str();
  if (cond) {
    // expected-note@-1 {{Assuming 'cond' is not equal to 0}}
    // expected-note@-2 {{Taking true branch}}
    // expected-note@-3 {{Assuming 'cond' is 0}}
    // expected-note@-4 {{Taking false branch}}
    consume(c1); // expected-warning {{Use of memory after it is freed}}
    // expected-note@-1 {{Use of memory after it is freed}}
  } else {
    consume(d1); // expected-warning {{Use of memory after it is freed}}
  } // expected-note@-1 {{Use of memory after it is freed}}
}

void deref_after_scope_cstr_ok() {
  const char *c;
  std::string s;
  {
    c = s.c_str();
  }
  consume(c); // no-warning
}

void deref_after_scope_data_ok() {
  const char *c;
  std::string s;
  {
    c = s.data();
  }
  consume(c); // no-warning
}
