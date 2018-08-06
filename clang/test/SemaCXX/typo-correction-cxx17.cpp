// RUN: %clang_cc1 -std=c++1z -fsyntax-only -verify %s

namespace decomp_decl {
void f() {
	auto [this_is_a_typo] = this_is_a_typp(); // expected-error{{use of undeclared identifier 'this_is_a_typp'}}
}
}
