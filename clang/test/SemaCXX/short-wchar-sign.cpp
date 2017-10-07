// RUN: %clang_cc1 -triple i386-mingw32 -fsyntax-only -pedantic -verify %s
// RUN: %clang_cc1 -fwchar-type=short -fno-signed-wchar -fsyntax-only -pedantic -verify %s
// RUN: %clang_cc1 -triple x86_64-apple-darwin10 -fsyntax-only -pedantic -verify %s
// expected-no-diagnostics

// Check that short wchar_t is unsigned, and that regular wchar_t is not.
int test[(wchar_t(-1)<wchar_t(0)) == (sizeof(wchar_t) == 4) ?1:-1];
