// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

// Test that the exclude_from_explicit_instantiation attribute is ignored
// for local classes and members thereof.

#define EXCLUDE_FROM_EXPLICIT_INSTANTIATION __attribute__((exclude_from_explicit_instantiation)) // expected-note 0+{{expanded from macro}}

namespace N0 {

  template<typename T>
  void f() {
    struct EXCLUDE_FROM_EXPLICIT_INSTANTIATION A { // expected-warning {{attribute ignored on local class}}
                                                   // expected-note@-1 2{{in instantiation of}}
      EXCLUDE_FROM_EXPLICIT_INSTANTIATION void g(T t) { // expected-warning {{attribute ignored on local class member}}
        *t; // expected-error {{indirection requires pointer operand ('int' invalid)}}
      }

      struct EXCLUDE_FROM_EXPLICIT_INSTANTIATION B { // expected-warning {{attribute ignored on local class}}
        void h(T t) {
          *t; // expected-error {{indirection requires pointer operand ('int' invalid)}}
        }
      };
    };
  }

  template void f<int>(); // expected-note 2{{in instantiation of}}

}

// This is a reduced example from libc++ which required that 'value'
// be prefixed with 'this->' because the definition of 'Local::operator A'
// was not instantiated when the definition of 'g' was.
namespace N1 {

  struct A { };

  struct B {
    operator A() {
      return A();
    }
  };

  template<typename T>
  auto f(T t) {
    return A(t);
  }

  template<typename T>
  auto g(T t) {
    struct Local {
      T value;

      EXCLUDE_FROM_EXPLICIT_INSTANTIATION // expected-warning {{attribute ignored on local class member}}
      operator A() {
        return A(value);
      }
    };

    return f(Local(t));
  }

  auto x = g(B());

}
