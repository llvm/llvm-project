// RUN: %clang_cc1  -fsyntax-only -verify %s
// RUN: %clang_cc1 -x objective-c++ -fsyntax-only -verify %s
// rdar://12456743

typedef signed char BOOL; // expected-note 2 {{candidate found by name lookup is 'BOOL'}}

EXPORT BOOL FUNC(BOOL enabled); // expected-error {{unknown type name 'EXPORT'}} // expected-error {{expected ';' after top level declarator}} \
                                // expected-note 2 {{candidate found by name lookup is 'BOOL'}}

static inline BOOL MFIsPrivateVersion(void) { // expected-error {{reference to 'BOOL' is ambiguous}}
 return __objc_yes; // expected-error {{reference to 'BOOL' is ambiguous}}
}
