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
