// RUN: %clang_cc1 -triple i686-pc-win32 -fms-extensions -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-NEG %s
// RUN: %clang_cc1 -triple i686-pc-win32 -fsyntax-only %s 2>&1 | FileCheck -check-prefix=CHECK-POS %s

// Do not report that 'foo()' is redeclared without dllimport attribute with -fms-extensions
// specified.  Addresses <rdar://problem/7653912>.

__declspec(dllimport) int __cdecl foo(void);
inline int __cdecl foo() { return 0; }

// This function is added just to trigger a diagnostic.  This way we can test how many
// diagnostics we expect.
void bar() { return 1; }

// CHECK-NEG: error: void function 'bar' should not return a value
// CHECK-NEG: 1 error generated
// CHECK-POS: warning: 'foo' redeclared without dllimport attribute: previous dllimport ignored
// CHECK-POS: error: void function 'bar' should not return a value
// CHECK-POS: 1 warning and 1 error generated

