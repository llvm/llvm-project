// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wno-signed-unsigned-wchar -verify=allow-signed -DSKIP_ERROR_TESTS %s
// allow-signed-no-diagnostics
wchar_t x;

void f(wchar_t p) {
  wchar_t x;
  unsigned wchar_t y; // expected-error {{'wchar_t' cannot be signed or unsigned}}
  signed wchar_t z; // expected-error {{'wchar_t' cannot be signed or unsigned}}
  ++x;
}

// PR4502
wchar_t const c = L'c';
int a[c == L'c' ? 1 : -1];


// PR5917
template<typename _CharT>
struct basic_string {
};

template<typename _CharT>
basic_string<_CharT> operator+ (const basic_string<_CharT>&, _CharT);

int t(void) {
  basic_string<wchar_t>() + L'-';
  return (0);
}


wchar_t in[] = L"\x434" "\x434";  // No warning

#ifndef SKIP_ERROR_TESTS
// Verify that we do not crash when assigning wchar_t* to another pointer type.
void assignment(wchar_t *x) {
  char *y;
  y = x; // expected-error {{incompatible pointer types assigning to 'char *' from 'wchar_t *'}}
}
#endif
