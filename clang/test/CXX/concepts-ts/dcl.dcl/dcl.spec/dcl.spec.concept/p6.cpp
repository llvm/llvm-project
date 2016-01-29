// RUN:  %clang_cc1 -std=c++14 -fconcepts-ts -x c++ -verify %s

template<typename T>
concept bool vc { true };

template<typename T>
struct B { typedef bool Boolean; };

template<int N>
B<void>::Boolean concept vctb(!0);

template<typename T>
concept const bool vctc { true }; // expected-error {{declared type of variable concept must be 'bool'}}

template<typename T>
concept int vcti { 5 }; // expected-error {{declared type of variable concept must be 'bool'}}

template<typename T>
concept float vctf { 5.5 }; // expected-error {{declared type of variable concept must be 'bool'}}

template<typename T>
concept auto vcta { true }; // expected-error {{declared type of variable concept must be 'bool'}}

template<typename T>
concept decltype(auto) vctd { true }; // expected-error {{declared type of variable concept must be 'bool'}}
