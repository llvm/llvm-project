// RUN: %clang_cc1 %s -std=c++14 -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify

// RUN: %clang_cc1 %s -std=c++14 -fsyntax-only -fexperimental-new-constant-interpreter -verify
// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -fexperimental-new-constant-interpreter -verify
// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -fexperimental-new-constant-interpreter -verify

template <typename T>
constexpr T foo(T a);   // expected-note {{declared here}}

int main() {
  int k = foo<int>(5);  // Ok
  constexpr int j =     // expected-error {{constexpr variable 'j' must be initialized by a constant expression}}
          foo<int>(5);  // expected-note {{undefined function 'foo<int>' cannot be used in a constant expression}}
}

template <typename T>
constexpr T foo(T a) {
  return a;
}

namespace GH73232 {
namespace ex1 {
template <typename T>
constexpr void g(T);

constexpr int f() {
  g(0);
  return 0;
}

template <typename T>
constexpr void g(T) {}

constexpr auto z = f();
}

namespace ex2 {
template <typename> constexpr static void fromType();

void registerConverter() { fromType<int>(); }
template <typename> struct QMetaTypeId  {};
template <typename T> constexpr void fromType() {
  (void)QMetaTypeId<T>{};
}
template <> struct QMetaTypeId<int> {};
} // namespace ex2

namespace ex3 {

#if __cplusplus > 202302L
struct A {
    consteval A(int i) {
        chk(i);
    }
    constexpr void chk(auto) {}
};
A a{1};

#endif

}

} // namespace GH73232


namespace GH156255 {

class X
{
public:
    constexpr int f( int x ) const
    {
        return g( x );
    }

private:

    template<class T>
    constexpr T g( T x ) const
    {
        return x;
    }
};

// check that g is instantiated here.
constexpr int x = X().f( 1 );
}

#if __cplusplus > 202002L

namespace instantiation_context_lookup {

static constexpr int i = 42;
static constexpr int v = 8;


constexpr int f(auto);

constexpr int g(int v = 42) {
    static constexpr int i = 1;
    return f(1);
    return 0;
}

constexpr int f(auto) {
    return i + v;
}

static_assert(g() == 50);

}

namespace GH35052 {

template <typename F>
constexpr int func(F f) {
    if constexpr (f(1UL)) {
        return 1;
    }
    return 0;
}

int test() {
    auto predicate = [](auto v) constexpr -> bool  { return v == 1; };
    return func(predicate); // check that "predicate" is instantiated.
}


}  // namespace GH35052

namespace GH115118 {

// Currently fails an assertion due to GH199347.
/*struct foo {
    foo(const foo&) = default;
    foo(auto)
        requires([]<int = 0>() -> bool { return true; }())
    {}
};

struct bar {
    foo x; // check that the lambda gets instantiated.
};*/

}  // namespace GH115118


namespace GH100897 {

template <typename>
constexpr auto foo() noexcept {
    constexpr auto extract_size = []<typename argument_t>() constexpr -> int {
        return 1;
    };

    constexpr int result = extract_size.template operator()<int>();
    return result;
}

void test() { foo<void>(); } // check that the lambda gets instantiated.

}  // namespace GH100897

namespace from_constexpr_initializer {
template <typename _CharT>
struct basic_string {
  constexpr void _M_construct();

  constexpr basic_string() {
    _M_construct();
  }

};

basic_string<char *> a;

template <typename _CharT>
constexpr void basic_string<_CharT>::_M_construct(){}

constexpr basic_string<char*> z{};
}  // namespace from_constexpr_initializer

namespace from_imm_invocation_in_immediate_escalating_fn {
template <int V> constexpr int f();
consteval int g() { return f<0>(); }
template <int V> constexpr int f() { return V; }

int h() { return [] { return g(); }(); }
}  // namespace from_imm_invocation_in_immediate_escalating_fn

namespace from_imm_invocation_in_non_escalating_fn {
template <int V> constexpr int f();
consteval int g() { return f<0>(); }
template <int V> constexpr int f() { return V; }

int h() { return g(); }
}  // namespace from_imm_invocation_in_non_escalating_fn

namespace from_template_argument {
template <int V> constexpr int f();
consteval int g() { return f<0>(); }
template <int V> constexpr int f() { return V; }

template <int V> consteval int h() { return V; }
int i() { return h<g()>(); }
}  // namespace from_template_argument

namespace from_constexpr_if {
template <int V> constexpr int f();
consteval int g() { return f<0>(); }
template <int V> constexpr int f() { return V; }

int h() {
  if constexpr (g())
    return 1;
  else
    return 2;
}
}  // namespace from_constexpr_if

namespace from_static_assertion {
template <int V> constexpr int f();
consteval int g() { return f<0>(); }
template <int V> constexpr int f() { return V; }

static_assert(g() == 0);
}  // namespace from_static_assertion

namespace from_constexpr_destructor {
template <int V> constexpr int f() noexcept;
struct S { constexpr ~S() { (void) f<0>(); } };
template <int V> constexpr int f() noexcept { return V; }

void h() { constexpr S s; }
}  // namespace from_constexpr_destructor

#endif
