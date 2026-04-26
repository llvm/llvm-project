// RUN: %clang_cc1 %s -verify -std=c++20

namespace std {

template<class T, class = T::x> // expected-error 2 {{type 'int' cannot be used prior to '::' because it has no members}}
class initializer_list;

}

namespace gh132256 {

auto x = {1}; // expected-note {{in instantiation of default argument for 'initializer_list<int>' required here}}

void f() {
	for(int x : {1, 2}); // expected-note {{in instantiation of default argument for 'initializer_list<int>' required here}}
}

}
