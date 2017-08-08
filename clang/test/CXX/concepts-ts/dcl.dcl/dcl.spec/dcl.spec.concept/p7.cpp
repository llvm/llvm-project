// RUN:  %clang_cc1 -std=c++14 -fconcepts-ts -x c++ -verify %s

template <typename T> concept bool FCEI() { return true; } // expected-note {{previous declaration is here}} expected-note {{previous declaration is here}}
template bool FCEI<int>(); // expected-error {{function concept cannot be explicitly instantiated}}
extern template bool FCEI<double>(); // expected-error {{function concept cannot be explicitly instantiated}}

template <typename T> concept bool FCES() { return true; } // expected-note {{previous declaration is here}}
template <> bool FCES<int>() { return true; } // expected-error {{function concept cannot be explicitly specialized}}

template <typename T> concept bool VC { true }; // expected-note {{previous declaration is here}} expected-note {{previous declaration is here}}
template bool VC<int>; // expected-error {{variable concept cannot be explicitly instantiated}}
extern template bool VC<double>; // expected-error {{variable concept cannot be explicitly instantiated}}

template <typename T> concept bool VCES { true }; // expected-note {{previous declaration is here}}
template <> bool VCES<int> { true }; // expected-error {{variable concept cannot be explicitly specialized}}

template <typename T> concept bool VCPS { true }; // expected-note {{previous declaration is here}}
template <typename T> bool VCPS<T *> { true }; // expected-error {{variable concept cannot be partially specialized}}
