// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

// These used to assert in Preprocessor::AnnotatePreviousCachedTokens instead
// of just diagnosing the error: ParseOptionalCXXScopeSpecifier annotated a
// decltype-specifier that failed to parse (no '(' after 'decltype'), using a
// stale end-location left over from error recovery.
int decltype = 0;
// expected-error@-1 {{expected '(' after 'decltype'}}
// expected-error@-2 {{expected unqualified-id}}

int *decltype = 0;
// expected-error@-1 {{expected '(' after 'decltype'}}
// expected-error@-2 {{expected unqualified-id}}
