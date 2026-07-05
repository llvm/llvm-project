// RUN: %clang_cc1 -triple i686-pc-win32 -fms-extensions -verify %s

// Do not report that 'foo()' is redeclared without dllimport attribute.
// specified.

// expected-no-diagnostics
__declspec(dllimport) int __cdecl foo(void);
inline int __cdecl foo() { return 0; }
