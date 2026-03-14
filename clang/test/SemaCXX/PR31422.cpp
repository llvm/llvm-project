// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify %s

// expected-error@+3 {{cannot specify any part of a return type in the declaration of a conversion function; use an alias template to declare a conversion to 'auto (Ts &&...) const'}}
// expected-error@+2 {{conversion function cannot convert to a function type}}
struct S {
  template <typename... Ts> operator auto()(Ts &&... xs) const;
};
