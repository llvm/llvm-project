// RUN:  %clang_cc1 -std=c++14 -fconcepts-ts -x c++ -verify %s

template<typename T>
concept bool fcpv(void) { return true; }

template<typename T>
concept bool fcpi(int i = 0) { return true; } // expected-error {{function concept cannot have any parameters}}

template<typename... Ts>
concept bool fcpp(Ts... ts) { return true; } // expected-error {{function concept cannot have any parameters}}

template<typename T>
concept bool fcpva(...) { return true; } // expected-error {{function concept cannot have any parameters}}

template<typename T>
concept const bool fcrtc() { return true; } // expected-error {{declared return type of function concept must be 'bool'}}

template<typename T>
concept int fcrti() { return 5; } // expected-error {{declared return type of function concept must be 'bool'}}

template<typename T>
concept float fcrtf() { return 5.5; } // expected-error {{declared return type of function concept must be 'bool'}}

template<typename T>
concept decltype(auto) fcrtd(void) { return true; } // expected-error {{declared return type of function concept must be 'bool'}}
