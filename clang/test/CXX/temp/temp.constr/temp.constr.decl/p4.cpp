// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics

template<typename T>
concept D = true;

template<typename T>
struct A {
  template<typename U, bool V>
  void f() requires V;

  template<>
  void f<short, true>();

  template<D U>
  void g();

  template<typename U, bool V> requires V
  struct B;

  template<typename U, bool V> requires V
  struct B<U*, V>;

  template<>
  struct B<short, true>;

  template<D U>
  struct C;

  template<D U>
  struct C<U*>;

  template<typename U, bool V> requires V
  static int x;

  template<typename U, bool V> requires V
  static int x<U*, V>;

  template<>
  int x<short, true>;

  template<D U>
  static int y;

  template<D U>
  static int y<U*>;
};

template<typename T>
template<typename U, bool V>
void A<T>::f() requires V { }

template<typename T>
template<D U>
void A<T>::g() { }

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
template<D U>
struct A<T>::C { };

template<typename T>
template<D U>
struct A<T>::C<U*> { };

template<typename T>
template<typename U, bool V> requires V
int A<T>::x = 0;

template<typename T>
template<typename U, bool V> requires V
int A<T>::x<U*, V> = 0;

template<typename T>
template<typename U, bool V> requires V
int A<T>::x<U&, V> = 0;

template<typename T>
template<D U>
int A<T>::y = 0;

template<typename T>
template<D U>
int A<T>::y<U*> = 0;

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
template<D U>
void A<short>::g();

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
template<D U>
struct A<int>::C;

template<>
template<D U>
struct A<int>::C<U*>;

template<>
template<D U>
struct A<int>::C<U&>;

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

template<>
template<D U>
int A<long>::y;

template<>
template<D U>
int A<long>::y<U*>;

template<>
template<D U>
int A<long>::y<U&>;
