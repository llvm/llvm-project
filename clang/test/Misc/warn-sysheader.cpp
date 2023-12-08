// Test that -Wsystem-headers works with default and custom mappings like -Werror.
// Keep run lines at the bottom for line number stability.

#ifdef IS_SYSHEADER
#pragma clang system_header

int f() { return (int)0; } // Use the old-style-cast warning as an arbitrary "ordinary" diagnostic for the purpose of testing.

#warning "custom message"

#if defined(A) || defined(B)
// expected-warning@9 {{"custom message"}}
#elif defined(C)
// expected-warning@7 {{use of old-style cast}}
// expected-warning@9 {{"custom message"}}
#elif defined(D)
// expected-error@7 {{use of old-style cast}}
// expected-error@9 {{"custom message"}}
#elif defined(E)
// expected-error@7 {{use of old-style cast}}
// expected-warning@9 {{"custom message"}}
#endif

#else
#define IS_SYSHEADER
#include __FILE__
#endif

// RUN: %clang_cc1 -verify -fsyntax-only -DA %s
// RUN: %clang_cc1 -verify -fsyntax-only -DB -Wold-style-cast %s
// RUN: %clang_cc1 -verify -fsyntax-only -DC -Wold-style-cast -Wsystem-headers %s
// RUN: %clang_cc1 -verify -fsyntax-only -DD -Wold-style-cast -Wsystem-headers -Werror %s
// RUN: %clang_cc1 -verify -fsyntax-only -DE -Wold-style-cast -Wsystem-headers -Werror=old-style-cast %s
