// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

// A 'decltype' that is not followed by '(' fails to parse as a
// decltype-specifier. ParseOptionalCXXScopeSpecifier used to still annotate it
// and treat it as a nested-name-specifier, which tripped an assertion in
// Preprocessor::AnnotatePreviousCachedTokens. It should just diagnose the
// error instead.
int decltype = 0;
// expected-error@-1 {{expected '(' after 'decltype'}}
// expected-error@-2 {{expected unqualified-id}}

int *decltype = 0;
// expected-error@-1 {{expected '(' after 'decltype'}}
// expected-error@-2 {{expected unqualified-id}}
