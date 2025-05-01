// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both %s

// both-no-diagnostics

namespace std {
template <class, int __v> struct integral_constant {
  static const int value = __v;
};
using size_t = decltype(sizeof(int));
template <class _Tp, class>
concept __weakly_equality_comparable_with = requires(_Tp __t) { __t; };
template <size_t, class> struct tuple_element;
template <class> struct tuple_size;
template <class _Ip>
concept input_or_output_iterator = requires(_Ip __i) { __i; };
template <class _Sp, class _Ip>
concept sentinel_for = __weakly_equality_comparable_with<_Sp, _Ip>;
namespace ranges {
enum subrange_kind { unsized };
template <input_or_output_iterator _Iter, sentinel_for<_Iter> _Sent,
          subrange_kind = unsized>
struct subrange {
  _Iter __begin_;
  _Sent __end_;
  constexpr _Sent end() { return __end_; }
};
template <int, class _Iter, class _Sent, subrange_kind _Kind>
constexpr auto get(subrange<_Iter, _Sent, _Kind> __subrange) {
  return __subrange.end();
}
} // namespace ranges
template <class _Ip, class _Sp, ranges::subrange_kind _Kp>
struct tuple_size<ranges::subrange<_Ip, _Sp, _Kp>>
    : integral_constant<size_t, 2> {};
template <class _Ip, class _Sp, ranges::subrange_kind _Kp>
struct tuple_element<0, ranges::subrange<_Ip, _Sp, _Kp>> {
  using type = _Ip;
};
template <class _Ip, class _Sp, ranges::subrange_kind _Kp>
struct tuple_element<1, ranges::subrange<_Ip, _Sp, _Kp>> {
  using type = _Sp;
};
} // namespace std
constexpr bool test() {
  int a[1];
  auto r = std::ranges::subrange(a, a);
  auto [first, last] = r;
  last = a;
  return true;
}
static_assert(test());

