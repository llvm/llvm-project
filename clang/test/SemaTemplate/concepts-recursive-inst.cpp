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

namespace DirectRecursiveCheck {
template<class T>
concept NotInf = true;
template<class T>
concept Inf = requires(T& v){ // #INF_REQ
  {begin(v)}; // #INF_BEGIN_EXPR
};

void begin(NotInf auto& v){ } // #NOTINF_BEGIN
// This lookup should fail, since it results in a recursive check.
// However, this is a 'hard failure'(not a SFINAE failure or constraints
// violation), so it needs to cause the entire lookup to fail.
void begin(Inf auto& v){ } // #INF_BEGIN

struct my_range{
} rng;

void baz() {
auto it = begin(rng); // #BEGIN_CALL
// expected-error@#INF_BEGIN {{satisfaction of constraint 'Inf<Inf auto>' depends on itself}}
// expected-note@#INF_BEGIN {{while substituting template arguments into constraint expression here}}
// expected-note@#INF_BEGIN_EXPR {{while checking constraint satisfaction for template 'begin<DirectRecursiveCheck::my_range>' required here}}
// expected-note@#INF_BEGIN_EXPR {{while substituting deduced template arguments into function template 'begin'}}
// expected-note@#INF_BEGIN_EXPR {{in instantiation of requirement here}}
// expected-note@#INF_REQ {{while substituting template arguments into constraint expression here}}
// expected-note@#INF_BEGIN {{while checking the satisfaction of concept 'Inf<DirectRecursiveCheck::my_range>' requested here}}
// expected-note@#INF_BEGIN {{while substituting template arguments into constraint expression here}}
// expected-note@#BEGIN_CALL {{while checking constraint satisfaction for template 'begin<DirectRecursiveCheck::my_range>' required here}}
// expected-note@#BEGIN_CALL {{in instantiation of function template specialization}}

// Fallout of the failure is failed lookup, which is necessary to stop odd
// cascading errors.
// expected-error@#BEGIN_CALL {{no matching function for call to 'begin'}}
// expected-note@#NOTINF_BEGIN {{candidate function}}
// expected-note@#INF_BEGIN{{candidate template ignored: constraints not satisfied}}
}
} // namespace DirectRecursiveCheck

namespace GH50891 {
  template <typename T>
  concept Numeric = requires(T a) { // #NUMERIC
      foo(a); // #FOO_CALL
    };

  struct Deferred {
    friend void foo(Deferred);
    template <Numeric TO> operator TO(); // #OP_TO
  };

  static_assert(Numeric<Deferred>); // #STATIC_ASSERT
  // expected-error@#NUMERIC{{satisfaction of constraint 'requires (T a) { foo(a); }' depends on itself}}
  // expected-note@#NUMERIC {{while substituting template arguments into constraint expression here}}
  // expected-note@#OP_TO {{while checking the satisfaction of concept 'Numeric<GH50891::Deferred>' requested here}}
  // expected-note@#OP_TO {{while substituting template arguments into constraint expression here}}
  // expected-note@#FOO_CALL {{while checking constraint satisfaction for template}}
  // expected-note@#FOO_CALL {{in instantiation of function template specialization}}
  // expected-note@#FOO_CALL {{in instantiation of requirement here}}
  // expected-note@#NUMERIC {{while substituting template arguments into constraint expression here}}

  // expected-error@#STATIC_ASSERT {{static assertion failed}}
  // expected-note@#STATIC_ASSERT{{while checking the satisfaction of concept 'Numeric<GH50891::Deferred>' requested here}}
  // expected-note@#STATIC_ASSERT{{because substituted constraint expression is ill-formed: constraint depends on a previously diagnosed expression}}

} // namespace GH50891


namespace GH60323 {
  // This should not diagnose, as it does not depend on itself.
  struct End {
        template<class T>
              void go(T t) { }

            template<class T>
                  auto endparens(T t)
                          requires requires { go(t); }
                { return go(t); }
  };

  struct Size {
        template<class T>
              auto go(T t)
                  { return End().endparens(t); }

            template<class T>
                  auto sizeparens(T t)
                          requires requires { go(t); }
                { return go(t); }
  };

  int f()
  {
        int i = 42;
            Size().sizeparens(i);
  }
}
