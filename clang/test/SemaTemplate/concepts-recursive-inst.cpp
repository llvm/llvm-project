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
// expected-error-re@#INF_REQ {{satisfaction of constraint {{.*}} depends on itself}}
// expected-note@#INF_BEGIN {{while checking the satisfaction of concept 'Inf<struct my_range>' requested here}}
// expected-note@#INF_BEGIN_EXPR {{while checking constraint satisfaction for template 'begin<struct my_range>' required here}}
// expected-note@#INF_BEGIN_EXPR {{while substituting deduced template arguments into function template 'begin'}}
// expected-note@#INF_BEGIN_EXPR {{in instantiation of requirement here}}
// expected-note@#INF_REQ {{while substituting template arguments into constraint expression here}}
// expected-note@#INF_BEGIN {{while checking the satisfaction of concept 'Inf<struct my_range>' requested here}}
// expected-note@#BEGIN_CALL {{while checking constraint satisfaction for template 'begin<struct my_range>' required here}}
// expected-note@#BEGIN_CALL {{while substituting deduced template arguments into function template}}

// Fallout of the failure is failed lookup, which is necessary to stop odd
// cascading errors.
// expected-error@#BEGIN_CALL {{no matching function for call to 'begin'}}
// expected-note@#NOTINF_BEGIN {{candidate function}}
// expected-note@#INF_BEGIN{{candidate template ignored: constraints not satisfied}}
// expected-note@#INF_BEGIN{{because 'Inf auto' does not satisfy 'Inf}}
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
  // expected-note@#OP_TO {{while checking the satisfaction of concept 'Numeric<Deferred>' requested here}}
  // expected-note@#OP_TO {{skipping 1 context}}
  // expected-note@#FOO_CALL 2{{while checking constraint satisfaction for template}}
  // expected-note@#FOO_CALL 2{{while substituting deduced template arguments into function template}}
  // expected-note@#FOO_CALL 2{{in instantiation of requirement here}}
  // expected-note@#NUMERIC {{while substituting template arguments into constraint expression here}}

  // expected-error@#STATIC_ASSERT {{static assertion failed}}
  // expected-note@#STATIC_ASSERT{{while checking the satisfaction of concept 'Numeric<Deferred>' requested here}}
  // expected-note@#STATIC_ASSERT{{because 'Deferred' does not satisfy 'Numeric'}}
  // expected-note@#FOO_CALL{{because 'foo(a)' would be invalid}}

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

namespace CWG2369_Regressions {

// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=109397
namespace GCC_103997 {

template<typename _type, typename _stream>
concept streamable = requires(_stream &s, _type &&v) {
  s << static_cast<_type &&>(v);
};

struct type_a {
  template<typename _arg>
  type_a &operator<<(_arg &&) {
    // std::clog << "type_a" << std::endl;
    return *this;
  }
};

struct type_b {
  type_b &operator<<(type_a const &) {
    // std::clog << "type_b" << std::endl;
    return *this;
  }
};

struct type_c {
  type_b b;
  template<typename _arg>
  requires streamable<_arg, type_b>
  friend type_c &operator<<(type_c &c, _arg &&a) {
    // std::clog << "type_c" << std::endl;
    c.b << static_cast<_arg &&>(a);
    return c;
  }
};

void foo() {
  type_a a;
  type_c c;
  a << c; // "type_a\n" (gcc gives error here)
  c << a; // "type_c\ntype_b\n"
}

}

// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=108393
namespace GCC_108393 {

template<class>
struct iterator_traits
{};

template<class T>
  requires requires(T __t, T __u) { __t == __u; }
struct iterator_traits<T>
{};

template<class T>
concept C = requires { typename iterator_traits<T>::A; };

struct unreachable_sentinel_t
{
  template<C _Iter>
  friend constexpr bool operator==(unreachable_sentinel_t, const _Iter&) noexcept;
};

template<class T>
struct S
{};

static_assert(!C<S<unreachable_sentinel_t>>);

}

// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=107429
namespace GCC_107429 {

struct tag_foo { } inline constexpr foo;
struct tag_bar { } inline constexpr bar;

template<typename... T>
auto f(tag_foo, T... x)
{
  return (x + ...);
}

template<typename... T>
concept fooable = requires (T... x) { f(foo, x...); };

template<typename... T> requires (fooable<T...>)
auto f(tag_bar, T... x)
{
  return f(foo, x...);
}

auto test()
{
  return f(bar, 1, 2, 3);
}

}

namespace GCC_99599 {

struct foo_tag {};
struct bar_tag {};

template <class T>
concept fooable = requires(T it) {
  invoke_tag(foo_tag{}, it); // <-- here
};

template <class T> auto invoke_tag(foo_tag, T in) { return in; }

template <fooable T> auto invoke_tag(bar_tag, T it) { return it; }

int main() {
  // Neither line below compiles in GCC 11, independently of the other
  return invoke_tag(foo_tag{}, 2) + invoke_tag(bar_tag{}, 2);
}

}

// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=99599#c22
namespace GCC_99599_2 {

template<typename T> class indirect {
public:
  template<typename U> requires
    requires (const T& t, const U& u) { t == u; }
  friend constexpr bool operator==(const indirect&, const U&) { return false; }

private:
  T* _M_ptr{};
};

indirect<int> i;
bool b = i == 1;

}

namespace GCC_99599_3 {

template<typename T>
struct S { T t; };

template<typename T>
concept C = sizeof(S<T>) > 0;

struct I;

struct from_range_t {
    explicit from_range_t() = default;
};
inline constexpr from_range_t from_range;

template<typename T>
concept FromRange = __is_same_as (T, from_range_t);

//#define WORKAROUND
#ifdef WORKAROUND
template<FromRange U, C T>
void f(U, T*);
#else
template<C T>
void f(from_range_t, T*);
#endif

void f(...);

void g(I* p) {
  f(0, p);
}

}

namespace GCC_99599_4 {

struct A {
  A(...);
};

template <class T> void f(A, T) { }

int main()
{
  f(42, 24);
}

}

namespace FAILED_GCC_110160 {
// https://gcc.gnu.org/bugzilla/show_bug.cgi?id=110160
// Current heuristic FAILED; GCC trunk also failed
// https://godbolt.org/z/r3Pz9Tehz
#if 0
#include <sstream>
#include <string>

template <class T>
concept StreamCanReceiveString = requires(T& t, std::string s) {
    { t << s };
};

struct NotAStream {};
struct UnrelatedType {};

template <StreamCanReceiveString S>
S& operator<<(S& s, UnrelatedType) {
    return s;
}

static_assert(!StreamCanReceiveString<NotAStream>);

static_assert(StreamCanReceiveString<std::stringstream>);
#endif
}
}
