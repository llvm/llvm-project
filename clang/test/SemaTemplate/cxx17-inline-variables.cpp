// RUN: %clang_cc1 -std=c++17 -verify %s
// expected-no-diagnostics
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
