// RUN: split-file %s %t
// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -verify %t/directive.cpp
// RUN: %clang_cc1 -fms-compatibility -fsyntax-only -verify %t/eof.cpp
// RUN: %clang_cc1 -fms-compatibility -fincremental-extensions -fsyntax-only -verify %t/eof.cpp

//--- directive.cpp
// A missing ')' must preserve the directive boundary (GH222310), so the
// following code is still parsed.
# 1 __identifier(foo // expected-error {{missing ')' after identifier}} expected-note {{to match this '('}}
int after_directive = undeclared; // expected-error {{use of undeclared identifier 'undeclared'}}

// Also check recovery when the directive is at the end of the file.
# 1 __identifier(foo // expected-error {{missing ')' after identifier}} expected-note {{to match this '('}}

//--- eof.cpp
// Keep this at the end of the file to test recovery at EOF and at the
// annotation marking the end of incremental input.
__identifier(foo // expected-error {{missing ')' after identifier}} expected-note {{to match this '('}}
