// RUN: %clang_cc1 -fsyntax-only -verify -std=c++20 %s

template<typename T>
int x;

template<typename T>
static int x<T*>;

template<>
static int x<int>; // expected-warning {{explicit specialization cannot have a storage class}}

template<typename T>
extern int y;

template<typename T>
static int y<T*>;

template<>
static int y<int>; // expected-warning {{explicit specialization cannot have a storage class}}

template<typename T>
void f();

template<>
static void f<int>(); // expected-warning {{explicit specialization cannot have a storage class}}

template<typename T>
extern void g();

template<>
static void g<int>(); // expected-warning {{explicit specialization cannot have a storage class}}

struct A {
  static int x;

  static int y;

  static void f();

  static void g();
};

int A::x = 0;

static int A::y = 0; // expected-error {{'static' can only be specified inside the class definition}}

void A::f() { }

static void A::g() { } // expected-error {{'static' can only be specified inside the class definition}}

struct B {
  template<typename T>
  static int x;

  template<typename T>
  static int y;

  template<typename T>
  int z; // expected-error {{non-static data member 'z' cannot be declared as a template}}

  template<typename T>
  static int x<T*>;

  template<typename T>
  static int y<T*>;

  template<typename T>
  int x<T**>; // expected-error {{non-static data member 'x' cannot be declared as a template}}

  template<>
  int x<short>;

  template<>
  static int x<long>; // expected-warning {{explicit specialization cannot have a storage class}}

  template<typename T>
  static void f();

  template<typename T>
  static void g();

  template<>
  void f<short>();

  template<>
  static void f<long>(); // expected-warning {{explicit specialization cannot have a storage class}}
};

template<typename T>
int B::x = 0;

template<typename T>
static int B::y = 0; // expected-error {{'static' can only be specified inside the class definition}}

template<typename T>
int B::x<T*> = 0;

template<typename T>
static int B::y<T*> = 0; // expected-error {{'static' can only be specified inside the class definition}}

template<typename T>
int B::x<T***>;

template<typename T>
static int B::y<T***>; // expected-error {{'static' can only be specified inside the class definition}}

template<>
int B::x<unsigned>;

template<>
static int B::y<unsigned>; // expected-warning {{explicit specialization cannot have a storage class}}
                           // expected-error@-1 {{'static' can only be specified inside the class definition}}

template<typename T>
void B::f() { }

template<typename T>
static void B::g() { } // expected-error {{'static' can only be specified inside the class definition}}

template<>
void B::f<unsigned>();

template<>
static void B::g<unsigned>(); // expected-warning {{explicit specialization cannot have a storage class}}
                              // expected-error@-1 {{'static' can only be specified inside the class definition}}

template<typename T>
struct C {
  static int x;

  static int y;

  static void f();

  static void g();
};

template<typename T>
int C<T>::x = 0;

template<typename T>
static int C<T>::y = 0; // expected-error {{'static' can only be specified inside the class definition}}

template<typename T>
void C<T>::f() { }

template<typename T>
static void C<T>::g() { } // expected-warning {{'static' can only be specified inside the class definition}}

template<>
int C<int>::x = 0;

template<>
static int C<int>::y = 0; // expected-warning {{explicit specialization cannot have a storage class}}
                          // expected-error@-1 {{'static' can only be specified inside the class definition}}

template<>
void C<int>::f();

template<>
static void C<int>::g(); // expected-warning {{explicit specialization cannot have a storage class}}
                         // expected-error@-1 {{'static' can only be specified inside the class definition}}
template<typename T>
struct D {
  template<typename U>
  static int x;

  template<typename U>
  static int y;

  template<typename U>
  int z; // expected-error {{non-static data member 'z' cannot be declared as a template}}

  template<typename U>
  static int x<U*>;

  template<typename U>
  static int y<U*>;

  template<typename U>
  int x<U**>; // expected-error {{non-static data member 'x' cannot be declared as a template}}

  template<>
  int x<short>;

  template<>
  static int x<long>; // expected-warning {{explicit specialization cannot have a storage class}}

  template<typename U>
  static void f();

  template<typename U>
  static void g();

  template<>
  void f<short>();

  template<>
  static void f<long>(); // expected-warning {{explicit specialization cannot have a storage class}}
};

template<typename T>
template<typename U>
int D<T>::x = 0;

template<typename T>
template<typename U>
static int D<T>::y = 0; // expected-error {{'static' can only be specified inside the class definition}}

template<typename T>
template<typename U>
int D<T>::x<U*> = 0;

template<typename T>
template<typename U>
static int D<T>::y<U*> = 0; // expected-error {{'static' can only be specified inside the class definition}}

template<typename T>
template<typename U>
int D<T>::x<U***>;

template<typename T>
template<typename U>
static int D<T>::y<U***>; // expected-error {{'static' can only be specified inside the class definition}}

template<>
template<typename U>
int D<int>::x;

template<>
template<typename U>
static int D<int>::y; // expected-warning {{explicit specialization cannot have a storage class}}
                      // expected-error@-1 {{'static' can only be specified inside the class definition}}
template<>
template<typename U>
int D<int>::x<U****>;

template<>
template<typename U>
static int D<int>::y<U****>; // expected-warning {{explicit specialization cannot have a storage class}}
                             // expected-error@-1 {{'static' can only be specified inside the class definition}}
template<>
template<>
int D<int>::x<unsigned>;

template<>
template<>
static int D<int>::y<unsigned>; // expected-warning {{explicit specialization cannot have a storage class}}
                                // expected-error@-1 {{'static' can only be specified inside the class definition}}

template<typename T>
template<typename U>
void D<T>::f() { }

template<typename T>
template<typename U>
static void D<T>::g() { } // expected-warning {{'static' can only be specified inside the class definition}}

template<>
template<typename U>
void D<int>::f();

template<>
template<typename U>
static void D<int>::g(); // expected-warning {{explicit specialization cannot have a storage class}}
                         // expected-error@-1 {{'static' can only be specified inside the class definition}}
template<>
template<>
void D<int>::f<unsigned>();

template<>
template<>
static void D<int>::g<unsigned>(); // expected-warning {{explicit specialization cannot have a storage class}}
                                   // expected-error@-1 {{'static' can only be specified inside the class definition}}
