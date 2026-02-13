// RUN: %clang_cc1 -std=c++17 -verify %s

template<bool> struct DominatorTreeBase {
  static constexpr bool IsPostDominator = true;
};
extern template class DominatorTreeBase<false>;
constexpr bool k = DominatorTreeBase<false>::IsPostDominator;

namespace CompleteType {
  template<unsigned N> constexpr int f(const bool (&)[N]) { return 0; }

  template<bool ...V> struct X {
    static constexpr bool arr[] = {V...};
    static constexpr int value = f(arr);
  };

  constexpr int n = X<true>::value;
}

template <typename T> struct A {
  static const int n;
  static const int m;
  constexpr int f() { return n; }
  constexpr int g() { return n; }
};
template <typename T> constexpr int A<T>::n = sizeof(A) + sizeof(T);
template <typename T> inline constexpr int A<T>::m = sizeof(A) + sizeof(T);
static_assert(A<int>().f() == 5);
static_assert(A<int>().g() == 5);

namespace GH135032 {

template <typename T> struct InlineAuto {
  template <typename G> inline static auto var = 5;
};

template <typename> struct PartialInlineAuto {
  template <typename, typename> inline static auto var = 6;
  template <typename T> inline static auto var<int, T> = 7;
};

int inline_auto = InlineAuto<int>::var<int>;
int partial_inline_auto = PartialInlineAuto<int>::var<int, int>;

}

namespace GH140773 {
template <class T> class ConstString { // #ConstString
  ConstString(typename T::type) {} // #ConstString-Ctor
};

template <class = int>
struct Foo {
  template <char>
  static constexpr ConstString kFilename{[] { // #kFileName
    return 42;
  }};
};

// We don't want to instantiate the member template until it's used!
Foo<> foo;

auto X = Foo<>::kFilename<'a'>;
// expected-error@#kFileName {{no viable constructor}}
// expected-note@-2 {{in instantiation of static data member}}
// expected-note@#ConstString-Ctor {{candidate template ignored}}
// expected-note@#ConstString-Ctor {{implicit deduction guide}}
// expected-note@#ConstString {{candidate template ignored}}
// expected-note@#ConstString {{implicit deduction guide}}

}
