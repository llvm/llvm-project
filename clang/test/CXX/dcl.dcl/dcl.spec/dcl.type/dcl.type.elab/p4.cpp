// RUN: %clang_cc1 -verify %s -std=c++11 -pedantic-errors

enum class E;

template<typename T>
struct A {
  enum class F;
};

struct B {
  template<typename T>
  friend enum A<T>::F; // expected-error {{elaborated enum specifier cannot be declared as a friend}}
                       // expected-note@-1 {{remove 'enum' to befriend an enum}}

  // FIXME: Per [temp.expl.spec]p19, a friend declaration cannot be an explicit specialization
  template<>
  friend enum A<int>::F; // expected-error {{elaborated enum specifier cannot be declared as a friend}}
                         // expected-note@-1 {{remove 'enum' to befriend an enum}}

  enum class G;

  friend enum E; // expected-error {{elaborated enum specifier cannot be declared as a friend}}
                 // expected-note@-1 {{remove 'enum' to befriend an enum}}
};

template<typename T>
struct C {
  friend enum T::G; // expected-error {{elaborated enum specifier cannot be declared as a friend}}
                    // expected-note@-1 {{remove 'enum' to befriend an enum}}
  friend enum A<T>::G; // expected-error {{elaborated enum specifier cannot be declared as a friend}}
                       // expected-note@-1 {{remove 'enum' to befriend an enum}}
};

struct D {
  friend enum B::G; // expected-error {{elaborated enum specifier cannot be declared as a friend}}
                    // expected-note@-1 {{remove 'enum' to befriend an enum}}
  friend enum class B::G; // expected-error {{elaborated enum specifier cannot be declared as a friend}}
                          // expected-note@-1 {{remove 'enum class' to befriend an enum}}
                          // expected-error@-2 {{reference to enumeration must use 'enum' not 'enum class'}}
};
