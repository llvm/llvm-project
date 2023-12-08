// RUN: %clang_cc1 %s -verify -fsyntax-only
- (void)compilerTestAgainst;  // expected-error {{missing context for method declaration}}

void xx();  // expected-error {{expected method body}}
