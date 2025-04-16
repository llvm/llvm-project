// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both %s

// both-no-diagnostics

namespace std {
inline namespace {
template <class _Tp, _Tp __v> struct integral_constant {
  static const _Tp value = __v;
};
template <bool _Val> using _BoolConstant = integral_constant<bool, _Val>;
template <class _Tp> using __remove_cv_t = __remove_cv(_Tp);
template <class _Tp> using remove_cv_t = __remove_cv_t<_Tp>;
} // namespace
inline namespace __1 {
template <class _Tp>
using __libcpp_remove_reference_t = __remove_reference_t(_Tp);
template <bool, class _IfRes, class> using conditional_t = _IfRes;
template <class _Tp, class _Up>
using _IsSame = _BoolConstant<__is_same(_Tp, _Up)>;
template <class _Tp> struct enable_if {
  typedef _Tp type;
};
template <bool, class _Tp = void> using __enable_if_t = _Tp;
template <class _Bp, class _Dp>
constexpr bool is_base_of_v = __is_base_of(_Bp, _Dp);
template <class _Tp> _Tp __declval(long);
template <class _Tp> decltype(__declval<_Tp>(0)) declval();
template <class _Fp, class... _Args>
constexpr decltype(declval<_Fp>()(declval<_Args>()...))
__invoke(_Fp __f, _Args... __args) {
  return (__f)((__args)...);
}
template <class, class _Fp, class... _Args> struct __invokable_r {
  template <class _XFp, class... _XArgs>
  static decltype(__invoke(declval<_XFp>(), declval<_XArgs>()...))
  __try_call(int);
  using _Result = decltype(__try_call<_Fp, _Args...>(0));
};
template <class _Func, class... _Args>
struct __invoke_result
    : enable_if<typename __invokable_r<void, _Func, _Args...>::_Result> {};
template <class _Fn, class... _Args>
using invoke_result_t = __invoke_result<_Fn, _Args...>::type;
template <class _Tp> constexpr __libcpp_remove_reference_t<_Tp> &&move(_Tp &&);
template <class _From, class _To>
constexpr bool is_convertible_v = __is_convertible(_From, _To);
template <class _From, class _To>
concept convertible_to =
    is_convertible_v<_From, _To> && requires { (declval<_From>()); };
template <class _Tp, class _Up>
concept __same_as_impl = _IsSame<_Tp, _Up>::value;
template <class _Tp, class _Up>
concept same_as = __same_as_impl<_Tp, _Up> && __same_as_impl<_Up, _Tp>;
template <class _Tp> using __remove_cvref_t = __remove_cvref(_Tp);
template <class _Tp> using remove_cvref_t = __remove_cvref_t<_Tp>;
template <class _Xp, class _Yp>
using __cond_res =
    decltype(false ? declval<_Xp (&)()>()() : declval<_Yp (&)()>()());
template <class...> struct common_reference;
template <class... _Types>
using common_reference_t = common_reference<_Types...>::type;
template <class, class> struct __common_reference_sub_bullet3;
template <class _Tp, class _Up>
struct common_reference<_Tp, _Up> : __common_reference_sub_bullet3<_Tp, _Up> {};
template <class _Tp, class _Up>
  requires requires { typename __cond_res<_Tp, _Up>; }
struct __common_reference_sub_bullet3<_Tp, _Up> {
  using type = __cond_res<_Tp, _Up>;
};
template <class _Tp, class _Up>
concept common_reference_with =
    same_as<common_reference_t<_Tp, _Up>, common_reference_t<_Up, _Tp>> &&
    convertible_to<_Tp, common_reference_t<_Tp, _Up>> &&
    convertible_to<_Up, common_reference_t<_Tp, _Up>>;
template <class _Tp>
using __make_const_lvalue_ref = __libcpp_remove_reference_t<_Tp>;
template <class _Lhs, class _Rhs>
concept assignable_from =
    common_reference_with<__make_const_lvalue_ref<_Lhs>,
                          __make_const_lvalue_ref<_Rhs>> &&
    requires(_Lhs __lhs, _Rhs __rhs) {
      { __lhs = (__rhs) };
    };
template <class _Tp>
concept default_initializable = requires { _Tp{}; };
template <class _Tp> constexpr bool is_object_v = __is_object(_Tp);
template <class _Dp, class _Bp>
concept derived_from = is_base_of_v<_Bp, _Dp> && is_convertible_v<_Dp *, _Bp *>;
template <class _Tp, class _Up>
concept __weakly_equality_comparable_with = requires(
    __make_const_lvalue_ref<_Tp> __t, __make_const_lvalue_ref<_Up> __u) {
  { __u };
};
template <class _Fn, class... _Args>
constexpr invoke_result_t<_Fn, _Args...> invoke(_Fn __f, _Args &&...__args) {
  return __invoke((__f), (__args)...);
}
template <class _Fn, class... _Args>
concept invocable =
    requires(_Fn __fn, _Args... __args) { invoke((__fn), (__args)...); };
template <class _Fn, class... _Args>
concept regular_invocable = invocable<_Fn, _Args...>;
template <template <class> class, class>
integral_constant<bool, false> __sfinae_test_impl();
template <template <class> class _Templ, class... _Args>
using _IsValidExpansion = decltype(__sfinae_test_impl<_Templ, _Args...>());
template <class _Tp>
using __test_for_primary_template =
    __enable_if_t<_IsSame<_Tp, typename _Tp::__primary_template>::value>;
template <class _Tp>
using __is_primary_template =
    _IsValidExpansion<__test_for_primary_template, _Tp>;
template <class> struct __cond_value_type;
template <class _Tp>
  requires is_object_v<_Tp>
struct __cond_value_type<_Tp> {
  using value_type = remove_cv_t<_Tp>;
};
template <class _Tp>
concept __has_member_value_type = requires { typename _Tp; };
template <class> struct indirectly_readable_traits;
template <class _Tp>
struct indirectly_readable_traits<_Tp *> : __cond_value_type<_Tp> {};
template <__has_member_value_type _Tp>
struct indirectly_readable_traits<_Tp>
    : __cond_value_type<typename _Tp::value_type> {};
template <bool> struct _OrImpl;
template <> struct _OrImpl<true> {
  template <class, class _First, class... _Rest>
  using _Result =
      _OrImpl<!bool() && sizeof...(_Rest)>::template _Result<_First, _Rest...>;
};
template <> struct _OrImpl<false> {
  template <class _Res> using _Result = _Res;
};
template <class... _Args>
using _Or =
    _OrImpl<sizeof...(_Args) !=
            0>::template _Result<integral_constant<bool, false>, _Args...>;
template <class _Tp>
concept __dereferenceable = requires(_Tp __t) {
  { __t };
};
template <__dereferenceable _Tp>
using iter_reference_t = decltype(*declval<_Tp>());
struct input_iterator_tag {};
struct forward_iterator_tag : input_iterator_tag {};
struct bidirectional_iterator_tag : forward_iterator_tag {};
struct __iter_concept_random_fallback {
  template <class>
  using _Apply = __enable_if_t<__is_primary_template<int>::value,
                               bidirectional_iterator_tag>;
};
template <class _Tester> struct __test_iter_concept : _Tester {};
struct __iter_concept_cache {
  using type = _Or<__test_iter_concept<__iter_concept_random_fallback>>;
};
template <class _Iter>
using _ITER_CONCEPT = __iter_concept_cache::type::_Apply<_Iter>;
template <class _Ip>
using iter_value_t =
    conditional_t<__is_primary_template<int>::value,
                  indirectly_readable_traits<remove_cvref_t<_Ip>>,
                  int>::value_type;
namespace ranges {
struct Trans_NS___iter_move___fn {
  template <class _Ip> auto operator()(_Ip __i) const -> decltype(std::move(*(__i)));
};
inline namespace {
auto iter_move = Trans_NS___iter_move___fn{};
}
} // namespace ranges
template <__dereferenceable _Tp>
  requires requires {
    { ranges::iter_move };
  }
using iter_rvalue_reference_t = decltype(ranges::iter_move(declval<_Tp>()));
template <class _In>
concept __indirectly_readable_impl =
    requires(_In __i) {
      { ranges::iter_move(__i) } -> same_as<iter_rvalue_reference_t<_In>>;
    } && common_reference_with<iter_reference_t<_In>, iter_value_t<_In>> &&
    common_reference_with<iter_reference_t<_In>,
                          iter_rvalue_reference_t<_In>> &&
    common_reference_with<iter_rvalue_reference_t<_In>, iter_value_t<_In>>;
template <class _In>
concept indirectly_readable = __indirectly_readable_impl<remove_cvref_t<_In>>;
template <class _Tp> struct __indirect_value_t_impl {
  using type = iter_value_t<_Tp>;
};
template <indirectly_readable _Tp>
using __indirect_value_t = __indirect_value_t_impl<_Tp>::type;
template <class _Sp, class _Ip>
concept sentinel_for = __weakly_equality_comparable_with<_Sp, _Ip>;
template <class _Ip>
concept input_iterator = requires { typename _ITER_CONCEPT<_Ip>; } &&
                         derived_from<_ITER_CONCEPT<_Ip>, input_iterator_tag>;
template <class _Fp, class _It>
concept indirectly_unary_invocable =
    invocable<_Fp, __indirect_value_t<_It>> &&
    invocable<_Fp, iter_reference_t<_It>> &&
    common_reference_with<invoke_result_t<_Fp, __indirect_value_t<_It>>,
                          invoke_result_t<_Fp, iter_reference_t<_It>>>;
template <class _Fp, class _It>
concept indirectly_regular_unary_invocable =
    regular_invocable<_Fp, __indirect_value_t<_It>> &&
    regular_invocable<_Fp, iter_reference_t<_It>> &&
    common_reference_with<invoke_result_t<_Fp, __indirect_value_t<_It>>,
                          invoke_result_t<_Fp, iter_reference_t<_It>>>;
template <class _Fp, class... _Its>
  requires(indirectly_readable<_Its> && ...) &&
              invocable<_Fp, iter_reference_t<_Its>...>
using indirect_result_t = invoke_result_t<_Fp, iter_reference_t<_Its>...>;
template <class _It, class _Proj> struct __projected_impl {
  struct __type {
    using value_type = remove_cvref_t<indirect_result_t<_Proj, _It>>;
    indirect_result_t<_Proj, _It> operator*();
  };
};
template <indirectly_readable _It,
          indirectly_regular_unary_invocable<_It> _Proj>
using projected = __projected_impl<_It, _Proj>::__type;
namespace ranges {
template <class, class> using for_each_result = int;

struct Z {};
constexpr Z zomg() {
  return {};
}
constexpr Z zomg2(Z z) {
  return {};
}

struct __for_each {

  template <input_iterator _Iter, sentinel_for<_Iter> _Sent, class _Proj,
            indirectly_unary_invocable<projected<_Iter, _Proj>> _Func>
  constexpr for_each_result<_Iter, _Func>
  operator()(_Iter __first, _Sent __last, _Func __func, _Proj __proj) const {

    for (; __first != __last; ++__first)
      invoke(__func, invoke(__proj, *__first));
    return {};
  }

};
inline namespace {
auto for_each = __for_each{};
}
} // namespace ranges
} // namespace __1
} // namespace std



struct T {};
struct Proj {
  constexpr void *operator()(T) { return nullptr; }
};
struct UnaryVoid {
  constexpr void operator()(void *) {}
};

constexpr bool all_the_algorithms() {
  T a[2];
  T *last = a + 2;
  std::ranges::for_each(a, last, UnaryVoid(), Proj());
  return true;
}
static_assert(all_the_algorithms());
