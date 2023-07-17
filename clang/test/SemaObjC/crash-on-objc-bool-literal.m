// RUN: %clang_cc1  -fsyntax-only -verify %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify %s
// rdar://12456743

typedef signed char BOOL;

EXPORT BOOL FUNC(BOOL enabled); // expected-error {{unknown type name 'EXPORT'}} // expected-error {{expected ';' after top level declarator}}

static inline BOOL MFIsPrivateVersion(void) {
 return __objc_yes;
}
