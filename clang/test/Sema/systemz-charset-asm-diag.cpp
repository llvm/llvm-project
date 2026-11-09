// REQUIRES: icu || iconv
// RUN: %clang_cc1 -triple s390x-none-zos -fexec-charset Shift_JIS %s -std=c++23 -fsyntax-only -verify

struct string_view {
  const char *D;
  int S;
  constexpr string_view(int Size, const char *Str) : D(Str), S(Size) {}
  constexpr const char *data() const { return D; }
  constexpr int size() const { return S; }
};

void asm_string_invalid_literal_encoding() {
  // The numeric escape "\x81" is stored unconverted in the literal encoding.
  // It is a Shift_JIS lead byte with no trail byte, so converting the evaluated
  // string back to UTF-8 fails.
  asm((string_view(1, "\x81"))); // expected-error-re {{conversion from literal encoding to UTF-8 failed: {{.*}}}}
}
