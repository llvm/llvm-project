// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

template<typename T>
struct A {
  template<typename U>
  struct B {
    static constexpr int y = 0;
  };

  template<typename U>
  struct B<U*> {
    static constexpr int y = 1;
  };

  template<typename U>
  static constexpr int x = 0;

  template<typename U>
  static constexpr int x<U*> = 1;
};

template<typename T>
template<typename U>
struct A<T>::B<U[]> {
  static constexpr int y = 2;
};

template<typename T>
template<typename U>
constexpr int A<T>::x<U[]> = 2;

static_assert(A<short>::B<int>::y == 0);
static_assert(A<short>::B<int*>::y == 1);
static_assert(A<short>::B<int[]>::y == 2);
static_assert(A<short>::x<int> == 0);
static_assert(A<short>::x<int*> == 1);
static_assert(A<short>::x<int[]> == 2);

template<>
template<typename U>
struct A<int>::B {
  static constexpr int y = 3;
};

template<>
template<typename U>
struct A<int>::B<U&> {
  static constexpr int y = 4;
};

template<>
template<typename U>
struct A<long>::B<U&> {
  static constexpr int y = 5;
};

template<>
template<typename U>
constexpr int A<int>::x = 3;

template<>
template<typename U>
constexpr int A<int>::x<U&> = 4;

template<>
template<typename U>
constexpr int A<long>::x<U&> = 5;

static_assert(A<int>::B<int>::y == 3);
static_assert(A<int>::B<int*>::y == 3);
static_assert(A<int>::B<int[]>::y == 3);

// FIXME: This should pass!
static_assert(A<int>::B<int&>::y == 4); // expected-error {{static assertion failed due to requirement 'A<int>::B<int &>::y == 4'}}
                                        // expected-note@-1 {{expression evaluates to '3 == 4'}}
static_assert(A<int>::x<int> == 3);
static_assert(A<int>::x<int*> == 3);
static_assert(A<int>::x<int[]> == 3);

// FIXME: This should pass!
static_assert(A<int>::x<int&> == 4); // expected-error {{static assertion failed due to requirement 'A<int>::x<int &> == 4'}}
                                     // expected-note@-1 {{expression evaluates to '3 == 4'}}
static_assert(A<long>::B<int>::y == 0);
static_assert(A<long>::B<int*>::y == 1);
static_assert(A<long>::B<int[]>::y == 2);
static_assert(A<long>::B<int&>::y == 5);
static_assert(A<long>::x<int> == 0);
static_assert(A<long>::x<int*> == 1);
static_assert(A<long>::x<int[]> == 2);
static_assert(A<long>::x<int&> == 5);
