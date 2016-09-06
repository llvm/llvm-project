// RUN:  %clang_cc1 -std=c++14 -fconcepts-ts -fcxx-exceptions -x c++ -verify %s

namespace A {
  template<typename T> concept bool C1() { return true; }

  template<typename T> concept bool C2 = true;
}

template<typename T> concept bool C3() { return (throw 0, true); }
static_assert(noexcept(C3<int>()), "function concept should be treated as if noexcept(true) specified");

template<typename T> concept bool D1(); // expected-error {{function concept declaration must be a definition}}

struct B {
  template<typename T> concept bool D2() { return true; } // expected-error {{concept declarations may only appear in namespace scope}}
};

struct C {
  template<typename T> static concept bool D3 = true; // expected-error {{concept declarations may only appear in namespace scope}}
};

concept bool D4() { return true; } // expected-error {{'concept' can only appear on the definition of a function template or variable template}}

concept bool D5 = true; // expected-error {{'concept' can only appear on the definition of a function template or variable template}}

template<typename T>
concept bool D6; // expected-error {{variable concept declaration must be initialized}}

template<typename T>
concept bool D7() throw(int) { return true; } // expected-error {{function concept cannot have exception specification}}

// Tag
concept class CC1 {}; // expected-error {{'concept' can only appear on the definition of a function template or variable template}}
concept struct CS1 {}; // expected-error {{'concept' can only appear on the definition of a function template or variable template}}
concept union CU1 {}; // expected-error {{'concept' can only appear on the definition of a function template or variable template}}
concept enum CE1 {}; // expected-error {{'concept' can only appear on the definition of a function template or variable template}}
template <typename T> concept class TCC1 {}; // expected-error {{'concept' can only appear on the definition of a function template or variable template}}
template <typename T> concept struct TCS1 {}; // expected-error {{'concept' can only appear on the definition of a function template or variable template}}
template <typename T> concept union TCU1 {}; // expected-error {{'concept' can only appear on the definition of a function template or variable template}}
typedef concept int CI; // expected-error {{'concept' can only appear on the definition of a function template or variable template}}
void fpc(concept int i) {} // expected-error {{'concept' can only appear on the definition of a function template or variable template}}

concept bool; // expected-error {{'concept' can only appear on the definition of a function template or variable template}}

template <typename T> concept bool VCEI{ true };
template concept bool VCEI<int>; // expected-error {{'concept' cannot be applied on an explicit instantiation}}
extern template concept bool VCEI<int>; // expected-error {{'concept' cannot be applied on an explicit instantiation}}

template <typename T> concept bool VCPS{ true };
template <typename T> concept bool VCPS<T *>{ true }; // expected-error {{'concept' cannot be applied on an partial specialization}}

template <typename T> concept bool VCES{ true };
template <> concept bool VCES<int>{ true }; // expected-error {{'concept' cannot be applied on an explicit specialization}}

template <typename T> concept bool FCEI() { return true; }
template concept bool FCEI<int>(); // expected-error {{'concept' cannot be applied on an explicit instantiation}}
extern template concept bool FCEI<int>(); // expected-error {{'concept' cannot be applied on an explicit instantiation}}

template <typename T> concept bool FCES() { return true; }
template <> concept bool FCES<bool>() { return true; } // expected-error {{'concept' cannot be applied on an explicit specialization}}
