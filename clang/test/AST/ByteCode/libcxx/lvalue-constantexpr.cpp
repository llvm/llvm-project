// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both                                             %s

// both-no-diagnostics

template <class _Tp, _Tp __v> struct integral_constant {
  static const _Tp value = __v;
};
template <bool _Val> using _BoolConstant = integral_constant<bool, _Val>;
namespace std {
template <class _Tp, class _Up>
using _IsSame = _BoolConstant<__is_same(_Tp, _Up)>;
template <class _Tp, class _Up>
concept __same_as_impl = _IsSame<_Tp, _Up>::value;
template <class _Tp, class _Up>
concept same_as = __same_as_impl<_Up, _Tp>;
template <class _Tp> _Tp forward;
template <class, class... _Args> struct __invoke_result_impl {
  using type = decltype(__builtin_invoke(_Args()...));
};
template <class... _Args>
using __invoke_result = __invoke_result_impl<void, _Args...>;
template <class... _Args>
using __invoke_result_t = __invoke_result<_Args...>::type;
template <class... _Args>
constexpr __invoke_result_t<_Args...> __invoke(_Args... __args) {
  return __builtin_invoke(__args...);
}
template <class _Fn, class... _Args>
using invoke_result_t = __invoke_result_t<_Fn, _Args...>;
template <class _Fn, class... _Args>
constexpr invoke_result_t<_Fn, _Args...> invoke(_Fn __f, _Args... __args) {
  return __invoke(__f, __args...);
}
template <class _Tp> struct __cw_fixed_value {
  constexpr __cw_fixed_value(_Tp) : __data() {}
  _Tp __data;
};
template <__cw_fixed_value> struct constant_wrapper;
template <__cw_fixed_value _Xp> auto cw = constant_wrapper<_Xp>{};
template <auto &_Callable, class... _Args>
concept __constexpr_callable = requires {
  typename constant_wrapper<invoke(_Callable, _Args ::value...)>;
};
template <__cw_fixed_value _Xp> struct constant_wrapper {
  static constexpr auto &value = _Xp.__data;
  void operator()(...);
  template <class... _Args>
    requires(!__constexpr_callable<value, _Args...>)
  auto operator()(_Args...) noexcept(invoke(value, forward<_Args>...)) {}
};
} // namespace std
template <class T> struct MustBeInt {
  static_assert(std::same_as<T, int>);
};

struct Poison {
  template <class T> constexpr auto operator()(T) -> MustBeInt<T> { return {}; }
};

bool test() {
  using T = std::constant_wrapper<Poison{}>;
  T()(std::cw<5>);
  return true;
}
