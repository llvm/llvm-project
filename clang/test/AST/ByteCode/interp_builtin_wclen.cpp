// RUN: %clang_cc1 -std=c++23 -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++23                                         -verify=ref,both %s

namespace WcslenInvalidArg {

  // 1) Narrow string literal passed to wcslen builtin: hard error in C++.
  static_assert(__builtin_wcslen("x") == 'x');
  // both-error@-1 {{cannot initialize a parameter of type 'const wchar_t *' with an lvalue of type 'const char[2]'}}

  // 2) Forced cast: should not crash, but not a constant expression.
  static_assert(__builtin_wcslen((const wchar_t *)"x") == 1);
  // both-error@-1 {{static assertion expression is not an integral constant expression}}
  // both-note@-2 {{cast that performs the conversions of a reinterpret_cast}}

  // 3) Forced cast from unsigned char*.
  const unsigned char u8s[] = "hi";
  static_assert(__builtin_wcslen((const wchar_t *)u8s) == 2);
  // both-error@-1 {{static assertion expression is not an integral constant expression}}
  // both-note@-2 {{cast that performs the conversions of a reinterpret_cast}}

  // 4) Correct wide string usage should constant-fold.
  static_assert(__builtin_wcslen(L"x") == 1);

} // namespace WcslenInvalidArg
