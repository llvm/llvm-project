// RUN: %clang_cc1 -std=c++17 -verify %s
// expected-no-diagnostics
template<bool> struct DominatorTreeBase {
  static constexpr bool IsPostDominator = true;
};
extern template class DominatorTreeBase<false>;
constexpr bool k = DominatorTreeBase<false>::IsPostDominator;
