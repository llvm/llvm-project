// RUN: %clang_cc1 -std=c++20 -w %s
// RUN: %clang_cc1 -std=c++2c -w %s
// expected-no-diagnostics

namespace std {
template <typename _Tp, _Tp __v> struct integral_constant {
  static constexpr _Tp value = __v;
  using value_type = _Tp;
};
template <bool __v> using __bool_constant = integral_constant<bool, __v>;
template <typename> struct is_integral : integral_constant<bool, true> {};
template <typename> struct is_signed : integral_constant<bool, false> {};
template <typename _Tp, typename _Up = _Tp> _Up __declval(int);
template <typename _Tp> auto declval() -> decltype(__declval<_Tp>(0));
template <typename> struct make_unsigned {
  using type = int;
};
template <typename _Tp> struct decay {
  using type = _Tp;
};
template <int, typename _Iftrue, typename> struct conditional {
  using type = _Iftrue;
};
} // namespace std
namespace meta {
template <template <typename...> class> struct quote;
template <template <typename> class C, typename... Ts>
concept valid = requires { typename C<Ts...>; };
template <typename T>
concept trait = requires { typename T; };
template <typename T>
concept invocable = requires { typename quote<T::template invoke>; };
template <typename T>
concept integral = requires { T::value; };
template <trait T> using _t = T::type;
template <integral T> constexpr T::value_type _v = T::value;
template <bool B> using bool_ = std::integral_constant<bool, B>;
template <invocable Fn, typename... Args>
using invoke = Fn::template invoke<Args...>;
template <typename> struct id;
namespace detail {
template <template <typename> class, typename...> struct defer_;
template <template <typename> class C, typename... Ts>
  requires valid<C, Ts...>
struct defer_<C, Ts...> {
  using type = C<Ts...>;
};
} // namespace detail
template <template <typename> class C, typename... Ts>
struct defer : detail::defer_<C, Ts...> {};
template <template <typename...> class C> struct quote {
  template <typename... Ts> using invoke = _t<defer<C, Ts...>>;
};
namespace detail {
template <int> struct _cond {
  template <typename Then, typename> using invoke = Then;
};
template <> struct _cond<false>;
} // namespace detail
template <bool If, typename Then, typename Else>
using conditional_t = detail::_cond<If>::template invoke<Then, Else>;
namespace detail {
template <typename...> struct _if_;
template <typename If, typename Then, typename Else>
struct _if_<If, Then, Else> : std::conditional<_v<If>, Then, Else> {};
} // namespace detail
template <bool If, typename... Args>
using if_c = _t<detail::_if_<bool_<If>, Args...>>;
} // namespace meta
template <bool> void requires_();
template <typename A, typename B>
concept same_as = __is_same(B, A);
namespace ranges {
template <typename> struct view_closure;
template <typename T> using decay_t = meta::_t<std::decay<T>>;
enum cardinality { unknown };
template <cardinality> struct basic_view {};
} // namespace ranges
namespace std {
template <typename> struct vector {};
} // namespace std
namespace ranges {
struct {
  template <typename F, typename... Args>
  auto operator()(F f, Args... args) -> decltype(f(args...));
} invoke;
template <typename Fun, typename... Args>
using invoke_result_t =
    decltype(invoke(std::declval<Fun>(), std::declval<Args>()...));
namespace detail {
struct with_difference_type_;
template <typename T> using iter_value_t_ = T ::value_type;
} // namespace detail
template <typename R> using iter_value_t = detail::iter_value_t_<R>;
namespace detail {
template <typename I>
using iter_size_t =
    meta::_t<meta::conditional_t<std::is_integral<I>::value,
                                 std::make_unsigned<I>, meta::id<I>>>;
template <typename D>
concept signed_integer_like_impl_concept_ =
    std::integral_constant<bool, -D()>::value;
template <typename D>
concept signed_integer_like_ = signed_integer_like_impl_concept_<D>;
} // namespace detail
template <typename S, typename I>
concept sized_sentinel_for_requires_ =
    requires(S s, I i) { requires_<same_as<I, decltype(i - s)>>; };
template <typename S, typename I>
concept sized_sentinel_for = sized_sentinel_for_requires_<S, I>;
struct range_access {
  template <typename Rng>
  static auto begin_cursor(Rng rng) -> decltype(rng.begin_cursor());
  template <typename Cur, typename O>
  static auto distance_to(Cur pos, O other) -> decltype(pos.distance_to(other));
};
namespace detail {
template <typename S, typename C>
concept sized_sentinel_for_cursor_requires_ = requires(S s, C c) {
  requires_<signed_integer_like_<decltype(range_access::distance_to(c, s))>>;
};
template <typename S, typename C>
concept sized_sentinel_for_cursor = sized_sentinel_for_cursor_requires_<S, C>;
struct iterator_associated_types_base_ {
  typedef range_access value_type;
};
template <typename>
using iterator_associated_types_base = iterator_associated_types_base_;
} // namespace detail
template <typename>
struct basic_iterator : detail::iterator_associated_types_base<int> {};
template <typename Cur2, typename Cur>
  requires detail::sized_sentinel_for_cursor<Cur2, Cur>
void operator-(basic_iterator<Cur2>, basic_iterator<Cur>);
namespace _begin_ {
template <typename T>
concept has_member_begin_requires_ = requires(T t) { t; };
template <typename T>
concept has_member_begin = has_member_begin_requires_<T>;
struct _member_result_ {
  template <typename R>
  using invoke = decltype(static_cast<R (*)()>(nullptr)().begin());
};
struct _non_member_result_;
struct fn {
  template <typename R>
  using _result_t =
      meta::invoke<meta::conditional_t<has_member_begin<R>, _member_result_,
                                       _non_member_result_>,
                   R>;
  template <typename R> _result_t<R> operator()(R);
};
} // namespace _begin_
_begin_::fn begin;
namespace _end_ {
template <typename>
concept has_member_end_requires_ = requires { begin; };
template <typename T>
concept has_member_end = has_member_end_requires_<T>;
struct _member_result_ {
  template <typename R>
  using invoke = decltype(static_cast<R (*)()>(nullptr)().end());
};
struct _non_member_result_;
struct fn {
  template <typename R>
  using _result_t =
      meta::invoke<meta::conditional_t<has_member_end<R>, _member_result_,
                                       _non_member_result_>,
                   R>;
  template <typename R> _result_t<R> operator()(R);
};
} // namespace _end_
_end_::fn end;
template <typename Rng>
using iterator_t = decltype(begin(static_cast<Rng (*)()>(nullptr)()));
template <typename Rng>
using sentinel_t = decltype(end(static_cast<Rng (*)()>(nullptr)()));
template <typename T>
concept has_member_size_requires_ = requires(T t) { t.size(); };
template <typename T>
concept has_member_size = has_member_size_requires_<T>;
struct _other_result_;
struct _member_result_ {
  template <typename> using invoke = decltype(0);
  template <typename R>
  using _result_t = meta::invoke<
      meta::conditional_t<has_member_size<R>, _member_result_, _other_result_>,
      R>;
  template <typename R> _result_t<R> operator()(R r) { r.size(); }
} size;
template <typename Rng> using range_value_t = iter_value_t<iterator_t<Rng>>;
namespace detail {
template <cardinality Card>
std::integral_constant<cardinality, Card> test_cardinality(basic_view<Card> *);
}
template <typename Rng>
struct range_cardinality
    : meta::conditional_t<__is_same(Rng, Rng),
                          decltype(detail::test_cardinality(
                              static_cast<Rng *>(nullptr))),
                          Rng> {};
template <typename T>
concept sized_range_requires_ = requires(T t) { size(t); };
template <typename T>
concept sized_range = sized_range_requires_<T>;
namespace detail {
template <int> struct dependent_ {
  template <typename T> using invoke = T;
};
} // namespace detail
template <typename Derived, cardinality Cardinality>
struct view_interface : basic_view<Cardinality> {
  template <bool B> using D = meta::invoke<detail::dependent_<B>, Derived>;
  Derived derived();
  template <bool True = true>
    requires sized_sentinel_for<sentinel_t<D<True>>, iterator_t<D<True>>>
  detail::iter_size_t<iterator_t<D<True>>> size() {
    derived().end() - derived().begin();
  }
};
struct {
  template <typename Fun> view_closure<Fun> operator()(Fun);
} make_view_closure;
struct view_closure_base {
  template <typename Rng, typename ViewFn>
  friend auto operator|(Rng rng, ViewFn vw) {
    return vw(rng);
  }
};
template <typename ViewFn> struct view_closure : view_closure_base, ViewFn {};
namespace detail {
template <typename Derived>
using begin_cursor_t =
    decay_t<decltype(range_access::begin_cursor(std::declval<Derived>()))>;
template <typename Derived>
using facade_iterator_t = basic_iterator<begin_cursor_t<Derived>>;
template <typename Derived>
using facade_sentinel_t =
    meta::if_c<same_as<Derived, Derived>, facade_iterator_t<Derived>, Derived>;
} // namespace detail
template <typename Derived, cardinality Cardinality>
struct view_facade : view_interface<Derived, Cardinality> {
  template <typename D = Derived> auto begin() -> detail::facade_iterator_t<D>;
  template <typename D = Derived> auto end() -> detail::facade_sentinel_t<D>;
};
template <typename Derived, cardinality Cardinality>
struct view_adaptor : view_facade<Derived, Cardinality> {
  auto begin_cursor() -> decltype(0);
};
namespace detail {
template <typename...> struct bind_back_fn_;
template <typename Fn, typename Arg> struct bind_back_fn_<Fn, Arg> {
  template <typename... CallArgs>
  invoke_result_t<Fn, CallArgs..., Arg> operator()(CallArgs...);
};
template <typename Fn, typename... Args>
using bind_back_fn = bind_back_fn_<Fn, Args...>;
} // namespace detail
struct {
  template <typename Fn, typename Arg1>
  detail::bind_back_fn<Fn, Arg1> operator()(Fn, Arg1);
} bind_back;
namespace detail {
struct to_container {
  template <typename> struct fn;
  template <typename, typename> struct closure;
};
template <typename, typename, typename R>
concept to_container_reserve = sized_range<R>;
template <typename MetaFn, typename Rng>
using container_t = meta::invoke<MetaFn, Rng>;
struct to_container_closure_base {
  template <typename Rng, typename MetaFn, typename Fn>
  friend auto operator|(Rng rng, to_container::closure<MetaFn, Fn> fn) {
    return fn(rng);
  }
};
template <typename, typename Fn>
struct to_container::closure : to_container_closure_base, Fn {};
template <typename MetaFn> struct to_container::fn {
  template <typename Rng> void impl(Rng, std::__bool_constant<false>);
  template <typename Rng> void impl(Rng rng, std::__bool_constant<true>) {
    size(rng);
  }
  template <typename Rng> container_t<MetaFn, Rng> operator()(Rng rng) {
    using cont_t = container_t<MetaFn, Rng>;
    using iter_t = Rng;
    using use_reserve_t =
        meta::bool_<to_container_reserve<cont_t, iter_t, Rng>>;
    impl(rng, use_reserve_t{});
  }
};
template <typename MetaFn, typename Fn>
using to_container_closure = to_container::closure<MetaFn, Fn>;
template <typename MetaFn>
using to_container_fn = to_container_closure<MetaFn, to_container::fn<MetaFn>>;
template <template <typename> class ContT> struct from_range {
  template <typename Rng>
  static auto from_rng_(long)
      -> meta::invoke<meta::quote<ContT>, range_value_t<Rng>>;
  template <typename Rng> using invoke = decltype(from_rng_<Rng>(0));
};
} // namespace detail
detail::to_container_fn<detail::from_range<std::vector>> to_vector;
template <typename Rng>
struct remove_if_view
    : view_adaptor<remove_if_view<Rng>, range_cardinality<Rng>::value> {};
struct filter_base_fn {
  template <typename Rng, typename Pred>
  remove_if_view<Rng> operator()(Rng, Pred);
  template <typename Pred> auto operator()(Pred pred) {
    return make_view_closure(bind_back(filter_base_fn{}, pred));
  }
} filter;
namespace detail {
struct promote_as_signed_;
template <typename I>
using iota_difference_t =
    meta::conditional_t<std::is_integral<I>::value, promote_as_signed_,
                        with_difference_type_>;
} // namespace detail
template <typename, typename>
struct iota_view : view_facade<iota_view<int, int>, unknown> {
  struct cursor {
    auto distance_to(cursor) -> detail::iota_difference_t<int>;
  };
  cursor begin_cursor();
};
struct {
  template <typename From, typename To>
    requires(std::is_signed<From>::value == std::is_signed<To>::value)
  iota_view<From, To> operator()(From, To);
} iota;
} // namespace ranges
void foo() {
  ranges::iota(0, 1) | ranges::to_vector =
      ranges::iota(0, 1) | ranges::filter([] {}) | ranges::to_vector;
}
