// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both                                             %s

// both-no-diagnostics

namespace std {
template <class _Tp> struct __cw_fixed_value {
  constexpr __cw_fixed_value(_Tp) : __data() {}
  _Tp __data;
};
template <__cw_fixed_value> struct constant_wrapper;
template <class _Tp>
concept __constexpr_param = requires { typename constant_wrapper<_Tp::value>; };
template <__cw_fixed_value _Xp> auto cw = constant_wrapper<_Xp>{};
struct __cw_operators {
  template <__constexpr_param _Lp, __constexpr_param _Rp>
  friend constexpr auto operator==(_Lp, _Rp) -> constant_wrapper<_Rp::value> {
    return {};
  }
};
template <__cw_fixed_value _Xp> struct constant_wrapper : __cw_operators {
  static constexpr auto value = _Xp.__data;
  constexpr operator decltype(value)() { return value; }
};
} // namespace std
void final_phase(auto gathered, auto available) {
  if constexpr (gathered == available)
    ;
}
void impeccable_underground_planning() {
  auto gathered_quantity(std::cw<3>), all_available = std::cw<5>;
  final_phase(gathered_quantity, all_available);
}


