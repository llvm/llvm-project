// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fno-recovery-ast -verify %s

namespace std {
template <typename>
class initializer_list{};
int a;
auto c = a, &d = {a}; // expected-error {{'auto' deduced as 'int'}} \
                         expected-error {{non-const lvalue reference to type}}
} // namespace std
