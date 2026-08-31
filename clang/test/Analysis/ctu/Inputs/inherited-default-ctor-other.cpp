namespace llvm {
template <int, typename...>
class impl;
// basecase
template <int n>
class impl<n> {};
// recursion
template <int n, typename T, typename... TS>
class impl<n, T, TS...> : impl<n + 1, TS...> {
  using child = impl<n + 1, TS...>;
  using child::child; // no-crash
  impl(T);
};
template <typename... TS>
class container : impl<0, TS...> {};
} // namespace llvm
namespace clang {
class fun {
  llvm::container<int, float> k;
  fun() {}
};
class DeclContextLookupResult {
  static int *const SingleElementDummyList;
};
} // namespace clang
using namespace clang;
int *const DeclContextLookupResult::SingleElementDummyList = nullptr;
