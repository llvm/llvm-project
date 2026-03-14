// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template<typename T>
struct X0 {
  struct MemberClass;

  T* f0(T* ptr);

  static T* static_member;
};

template class X0<int(int)>; // ok; nothing gets instantiated.

template<typename T>
struct X0<T>::MemberClass {
  T member;
};

template<typename T>
T* X0<T>::f0(T* ptr) {
  return ptr + 1;
}

template<typename T>
T* X0<T>::static_member = 0;

template class X0<int>; // ok


template<typename T>
struct X1 {
  enum class E {
    e = T::error // expected-error 2{{no members}}
  };
};
template struct X1<int>; // expected-note {{here}}

extern template struct X1<char>; // ok

template struct X1<char>; // expected-note {{here}}
