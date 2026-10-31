// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both                                             %s

// both-no-diagnostics

namespace std {

class SomeBase {
public:
  int k = 20;
};

template <class, class... _Args>
constexpr bool is_nothrow_invocable_v = noexcept(__builtin_invoke(_Args()...));
template <class _Tp> struct __cw_fixed_value : public SomeBase {
  constexpr __cw_fixed_value(_Tp) {}
  _Tp __data = 0;
  int a[3]{1,1,1};
};
template <__cw_fixed_value> struct constant_wrapper;
template <auto &_Callable>
concept __constexpr_callable =
    requires { typename constant_wrapper<_Callable>; };
template <__cw_fixed_value _Xp> struct constant_wrapper {
  static constexpr auto &value = _Xp.__data;
  static constexpr auto &value2 = _Xp.a[1];
  static constexpr auto &value3 = _Xp.k;
  template <class...>
    requires __constexpr_callable<value> && __constexpr_callable<value2> && __constexpr_callable<value3>
  constant_wrapper operator()() noexcept;
};
} // namespace std
void throwing_call();
static_assert(
    std::is_nothrow_invocable_v<std::constant_wrapper<throwing_call>,
                                std::constant_wrapper<42>>,
    "the call expression is still nothrow because the constexpr path is taken");


