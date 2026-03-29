// RUN: %clang_cc1 -fsyntax-only -verify %s

// expected-error@10 {{template template parameter requires 'class' or 'typename' after the parameter list}}
// expected-error@10 {{template template parameter must have its own template parameters}}
// expected-error@10 {{no template named 'a' in the global namespace}}
// expected-note@10 {{to match this '<'}}
// expected-error@10 {{expected expression}}
// expected-error@10 {{expected '>'}}
// expected-error@10 {{expected unqualified-id}}
template <template <> a>::a <
