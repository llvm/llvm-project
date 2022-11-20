// RUN: %clang_cc1 -std=c++20 -verify %s
namespace GH53213 {
template<typename T>
concept c = requires(T t) { f(t); }; // #CDEF

auto f(c auto); // #FDEF

void g() {
  f(0);
  // expected-error@-1{{no matching function for call to 'f'}}
  // expected-note@#FDEF{{constraints not satisfied}}
  // expected-note@#FDEF{{because 'int' does not satisfy 'c'}}
  // expected-note@#CDEF{{because 'f(t)' would be invalid: no matching function for call to 'f'}}
}
} // namespace GH53213 

namespace GH45736 {
struct constrained;

template<typename T>
  struct type {
  };
template<typename T>
  constexpr bool f(type<T>) {
      return true;
  }

template<typename T>
  concept matches = f(type<T>());


struct constrained {
    template<typename U> requires matches<U>
        explicit constrained(U value) {
            }
};

bool f(constrained const &) {
    return true;
}

struct outer {
    constrained state;
};

bool f(outer const & x) {
    return f(x.state);
}
} // namespace GH45736
