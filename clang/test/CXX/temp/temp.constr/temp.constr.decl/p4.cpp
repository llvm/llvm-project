// RUN: %clang_cc1 -std=c++20 -verify %s

template<typename T>
struct A {
  template<typename U, bool V>
  void f() requires V;

  template<>
  void f<short, true>();

  template<typename U, bool V> requires V
  struct B;

  template<typename U, bool V> requires V
  struct B<U*, V>;

  template<>
  struct B<short, true>;

  template<typename U, bool V> requires V
  static int x;

  template<typename U, bool V> requires V
  static int x<U*, V>;

  template<>
  int x<short, true>;
};

template<typename T>
template<typename U, bool V>
void A<T>::f() requires V { }

template<typename T>
template<typename U, bool V> requires V
struct A<T>::B { };

template<typename T>
template<typename U, bool V> requires V
struct A<T>::B<U*, V> { };

template<typename T>
template<typename U, bool V> requires V
struct A<T>::B<U&, V> { };

template<typename T>
template<typename U, bool V> requires V
int A<T>::x = 0;

template<typename T>
template<typename U, bool V> requires V
int A<T>::x<U*, V> = 0;

template<typename T>
template<typename U, bool V> requires V
int A<T>::x<U&, V> = 0;

template<>
template<typename U, bool V>
void A<short>::f() requires V;

template<>
template<>
void A<short>::f<int, true>();

template<>
template<>
void A<void>::f<int, true>();

template<>
template<typename U, bool V> requires V
struct A<int>::B;

template<>
template<>
struct A<int>::B<int, true>;

template<>
template<>
struct A<void>::B<int, true>;

template<>
template<typename U, bool V> requires V
struct A<int>::B<U*, V>;

template<>
template<typename U, bool V> requires V
struct A<int>::B<U&, V>;

template<>
template<typename U, bool V> requires V
int A<long>::x;

template<>
template<>
int A<long>::x<int, true>;

template<>
template<>
int A<void>::x<int, true>;

template<>
template<typename U, bool V> requires V
int A<long>::x<U*, V>;

template<>
template<typename U, bool V> requires V
int A<long>::x<U&, V>;
