// RUN: %clang_cc1 %s -std=c++14 -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -verify
// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -verify

// RUN: %clang_cc1 %s -std=c++14 -fsyntax-only -fexperimental-new-constant-interpreter -verify
// RUN: %clang_cc1 %s -std=c++20 -fsyntax-only -fexperimental-new-constant-interpreter -verify
// RUN: %clang_cc1 %s -std=c++2c -fsyntax-only -fexperimental-new-constant-interpreter -verify

template <typename T>
constexpr T foo(T a);   // expected-note {{explicit instantiation refers here}}

int main() {
  int k = foo<int>(5);  // Ok
  constexpr int j =
          foo<int>(5);  // expected-error {{explicit instantiation of undefined function template 'foo'}} \
                        // expected-error {{constexpr variable 'j' must be initialized by a constant expression}}
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

constexpr int x = X().f( 1 );
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
    return func(predicate);
}

}  // namespace GH35052

namespace GH115118 {

struct foo {
    foo(const foo&) = default;
    foo(auto)
        requires([]<int = 0>() -> bool { return true; }())
    {}
};

struct bar {
    foo x;
};

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

void test() { foo<void>(); }

}  // namespace GH100897
