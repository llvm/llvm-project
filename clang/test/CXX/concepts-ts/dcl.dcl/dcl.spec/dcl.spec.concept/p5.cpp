// RUN:  %clang_cc1 -std=c++14 -fconcepts-ts -x c++ -verify %s

template<typename T>
concept bool fcpv(void) { return true; }

template<typename T>
concept bool fcpi(int i = 0) { return true; } // expected-error {{function concept cannot have any parameters}}

template<typename... Ts>
concept bool fcpp(Ts... ts) { return true; } // expected-error {{function concept cannot have any parameters}}

template<typename T>
concept bool fcpva(...) { return true; } // expected-error {{function concept cannot have any parameters}}
