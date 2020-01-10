// RUN:  %clang_cc1 -std=c++2a -fconcepts-ts -verify %s

auto l1 = [] (auto x) requires (sizeof(decltype(x)) == 1) { return x; };
// expected-note@-1{{candidate template ignored: constraints not satisfied [with $0 = int]}}
// expected-note@-2{{because 'sizeof(decltype(x)) == 1' (4 == 1) evaluated to false}}

auto l1t1 = l1('a');
auto l1t2 = l1(1);
// expected-error@-1{{no matching function for call to object of type '(lambda at}}

auto l2 = [] (auto... x) requires ((sizeof(decltype(x)) >= 2) && ...) { return (x + ...); };
// expected-note@-1{{candidate template ignored: constraints not satisfied [with $0 = <char>]}}
// expected-note@-2{{candidate template ignored: constraints not satisfied [with $0 = <int, char>]}}
// expected-note@-3 2{{because 'sizeof(decltype(x)) >= 2' (1 >= 2) evaluated to false}}

auto l2t1 = l2('a');
// expected-error@-1{{no matching function for call to object of type '(lambda at}}
auto l2t2 = l2(1, 'a');
// expected-error@-1{{no matching function for call to object of type '(lambda at}}
auto l2t3 = l2((short)1, (short)1);