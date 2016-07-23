// RUN: %clang_cc1 -std=c++1z -verify %s

void use_from_own_init() {
  auto [a] = a; // expected-error {{binding 'a' cannot appear in the initializer of its own decomposition declaration}}
}

// FIXME: create correct bindings
// FIXME: template instantiation
// FIXME: ast file support
// FIXME: code generation
// FIXME: constant expression evaluation
