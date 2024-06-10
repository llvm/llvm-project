// RUN: %clang_cc1 -std=c++20 %s -emit-llvm -o /dev/null -verify -triple %itanium_abi_triple
// RUN: %clang_cc1 -std=c++20 %s -emit-llvm -o /dev/null -verify -triple %ms_abi_triple

// expected-no-diagnostics

struct BasicPersistent;
template <typename> BasicPersistent &&__declval(int);
template <typename _Tp> auto declval() -> decltype(__declval<_Tp>(0));
template <typename _Tp> _Tp forward;
template <typename _Tp, typename... _Args>
auto construct_at(_Tp *, _Args...) -> decltype(new _Tp(declval<_Args>()...)) {return 0;}
template <typename> struct allocator;
template <typename> struct allocator_traits;
template <typename _Tp> struct allocator_traits<allocator<_Tp>> {
  using pointer = _Tp *;
  template <typename _Up, typename... _Args>
  static void construct(_Up __p, _Args...) {
    construct_at(__p, forward<_Args>...);
  }
};
struct __alloc_traits : allocator_traits<allocator<BasicPersistent>> {
} push_back___x;
__alloc_traits::pointer _M_impl_0;
template <typename... _Args> void emplace_back(_Args...) {
  __alloc_traits::construct(_M_impl_0, forward<_Args>...);
}
struct SourceLocation {
  static SourceLocation Current(const char * = __builtin_FUNCTION());
};
struct BasicPersistent {
  BasicPersistent(BasicPersistent &&,
                  SourceLocation = SourceLocation::Current());
};
void CFXJSE_EngineAddObjectToUpArray() { emplace_back(push_back___x); }
