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
template <class _From, class _To>
constexpr bool is_convertible_v = __is_convertible(_From, _To);
template <class _Tp> _Tp __declval(long);
template <class _Tp> decltype(__declval<_Tp>(0)) declval();
template <class _From, class _To>
concept convertible_to = is_convertible_v<_From, _To> &&
                         requires { static_cast<_To>(declval<_From>()); };
template <class _Tp> constexpr bool is_reference_v = __is_reference(_Tp);
template <class _Tp>
constexpr bool is_lvalue_reference_v = __is_lvalue_reference(_Tp);
template <class>
constexpr bool is_nothrow_destructible_v =
    integral_constant<bool, __is_nothrow_destructible(int)>::value;
template <class _Tp>
concept destructible = is_nothrow_destructible_v<_Tp>;
template <class _Tp, class _Up>
using _IsSame = _BoolConstant<__is_same(_Tp, _Up)>;
template <class... _Args>
constexpr bool is_constructible_v = __is_constructible(_Args...);
template <class _Tp, class... _Args>
concept constructible_from = destructible<_Tp> && is_constructible_v<_Tp>;
template <class _Tp>
concept move_constructible =
    constructible_from<_Tp, _Tp> && convertible_to<_Tp, _Tp>;
template <class _Tp, class _Up>
concept __same_as_impl = _IsSame<_Tp, _Up>::value;
template <class _Tp, class _Up>
concept same_as = __same_as_impl<_Tp, _Up> && __same_as_impl<_Up, _Tp>;
template <bool> struct _IfImpl;
template <> struct _IfImpl<false> {
  template <class, class _ElseRes> using _Select = _ElseRes;
};
template <bool _Cond, class _IfRes, class _ElseRes>
using _If = _IfImpl<_Cond>::template _Select<_IfRes, _ElseRes>;
template <class _If> struct conditional {
  using type = _If;
};
template <bool, class _IfRes, class>
using conditional_t = conditional<_IfRes>::type;
template <class _Tp>
using __libcpp_remove_reference_t = __remove_reference_t(_Tp);
template <class _Tp>
using remove_reference_t = __libcpp_remove_reference_t<_Tp>;
template <class _Tp> using __decay_t = __decay(_Tp);
template <class _Tp> using __remove_cvref_t = __remove_cvref(_Tp);
template <class _Tp> using remove_cvref_t = __remove_cvref_t<_Tp>;
struct __copy_cv {
  template <class _To> using __apply = _To;
};
template <class, class _To> using __copy_cv_t = __copy_cv::__apply<_To>;
template <class _Xp, class _Yp>
using __cond_res =
    decltype(false ? std::declval<_Xp (&)()>()() : std::declval<_Yp (&)()>()());
template <class _Ap, class _Bp, class = remove_reference_t<_Ap>,
          class = remove_reference_t<_Bp>>
struct __common_ref;
template <class _Ap, class _Bp, class, class>
struct __common_ref : __common_ref<_Bp, _Ap> {};
template <class _Xp, class _Yp>
using __common_ref_t = __common_ref<_Xp, _Yp>::__type;
template <class _Xp, class _Yp>
using __cv_cond_res =
    __cond_res<__copy_cv_t<_Xp, _Yp> &, __copy_cv_t<_Yp, _Xp> &>;
template <class _Ap, class _Bp, class _Xp, class _Yp>
  requires requires { typename __cv_cond_res<_Xp, _Yp>; } &&
           is_reference_v<__cv_cond_res<_Xp, _Yp>>
struct __common_ref<_Ap, _Bp &, _Xp, _Yp> {
  using __type = __cv_cond_res<_Xp, _Yp>;
};
template <class _Tp, class _Up>
using __common_ref_D = __common_ref_t<const _Tp, _Up &>;
template <class _Ap, class _Bp, class _Xp, class _Yp>
  requires requires { typename __common_ref_D<_Xp, _Yp>; } &&
           is_convertible_v<_Ap, __common_ref_D<_Xp, _Yp>>
struct __common_ref<_Ap &&, _Bp &, _Xp, _Yp> {
  using __type = __common_ref_D<_Xp, _Yp>;
};
template <class...> struct common_reference;
template <class... _Types>
using common_reference_t = common_reference<_Types...>::type;
template <class, class> struct __common_reference_sub_bullet1;
template <class _Tp, class _Up>
struct common_reference<_Tp, _Up> : __common_reference_sub_bullet1<_Tp, _Up> {};
template <class _Tp, class _Up>
  requires is_reference_v<_Tp> && is_reference_v<_Up> &&
           requires { typename __common_ref_t<_Tp, _Up>; }
struct __common_reference_sub_bullet1<_Tp, _Up> {
  using type = __common_ref_t<_Tp, _Up>;
};
template <class _Tp, class _Up>
concept common_reference_with =
    same_as<common_reference_t<_Tp, _Up>, common_reference_t<_Up, _Tp>> &&
    convertible_to<_Tp, common_reference_t<_Tp, _Up>> &&
    convertible_to<_Up, common_reference_t<_Tp, _Up>>;
template <class _Tp>
using __make_const_lvalue_ref = __libcpp_remove_reference_t<_Tp> &;
template <class _Lhs, class _Rhs>
concept assignable_from =
    is_lvalue_reference_v<_Lhs> &&
    common_reference_with<__make_const_lvalue_ref<_Lhs>,
                          __make_const_lvalue_ref<_Rhs>> &&
    requires(_Lhs __lhs, _Rhs __rhs) {
      { __rhs } -> same_as<_Lhs>;
    };
template <class _Tp> constexpr __libcpp_remove_reference_t<_Tp> &&move(_Tp &&);
typedef int type;
template <bool, class = void> using __enable_if_t = type;
namespace ranges {
inline namespace {
auto swap = int{};
}
} // namespace ranges
template <class _Tp> constexpr bool is_object_v = __is_object(_Tp);
template <class _Tp>
concept movable = is_object_v<_Tp> && move_constructible<_Tp> &&
                  assignable_from<_Tp &, _Tp>;
template <decltype(sizeof(int)), class> struct tuple_element;
template <class...> class tuple;
template <template <class> class _Templ, class... _Args,
          class = _Templ<_Args...>>
integral_constant<bool, true> __sfinae_test_impl(int);
template <template <class> class, class>
integral_constant<bool, false> __sfinae_test_impl(...);
template <template <class> class _Templ, class... _Args>
using _IsValidExpansion =
    decltype(std::__sfinae_test_impl<_Templ, _Args...>(0));
template <class _Tp>
using __test_for_primary_template =
    __enable_if_t<_IsSame<_Tp, typename _Tp::__primary_template>::value>;
template <class _Tp>
using __is_primary_template =
    _IsValidExpansion<__test_for_primary_template, _Tp>;
template <class> struct iterator_traits;
template <class> struct __cond_value_type;
template <class _Tp>
  requires is_object_v<_Tp>
struct __cond_value_type<_Tp> {
  using value_type = remove_cv_t<_Tp>;
};
template <class> struct indirectly_readable_traits;
template <class _Tp>
struct indirectly_readable_traits<_Tp *> : __cond_value_type<_Tp> {};
template <bool> struct _OrImpl;
template <> struct _OrImpl<true> {
  template <class, class _First, class... _Rest>
  using _Result = _OrImpl<!(_First::value) &&
                          sizeof...(_Rest)>::template _Result<_First, _Rest...>;
};
template <> struct _OrImpl<false> {
  template <class _Res, class...> using _Result = _Res;
};
template <class... _Args>
using _Or =
    _OrImpl<sizeof...(_Args) !=
            0>::template _Result<integral_constant<bool, false>, _Args...>;
template <class _Tp> using __with_reference = _Tp;
template <class _Tp>
concept __can_reference = requires { typename __with_reference<_Tp>; };
template <class _Tp>
concept __dereferenceable = requires(_Tp __t) {
  { __t };
};
template <__dereferenceable _Tp>
using iter_reference_t = decltype(*std::declval<_Tp>());
struct input_iterator_tag {};
struct contiguous_iterator_tag : input_iterator_tag {};
template <class> struct __iter_traits_cache {
  using type = _If<__is_primary_template<iterator_traits<int>>::value, int,
                   iterator_traits<int>>;
};
template <class _Iter> using _ITER_TRAITS = __iter_traits_cache<_Iter>::type;
struct __iter_concept_concept_test {
  template <class _Iter> using _Apply = _ITER_TRAITS<_Iter>::iterator_concept;
};
template <class, class _Tester>
struct __test_iter_concept : _IsValidExpansion<_Tester::template _Apply, int>,
                             _Tester {};
template <class _Iter> struct __iter_concept_cache {
  using type =
      _Or<__test_iter_concept<_Iter, __iter_concept_concept_test>,
          __test_iter_concept<_Iter, int>, __test_iter_concept<_Iter, int>>;
};
template <class _Iter>
using _ITER_CONCEPT = __iter_concept_cache<_Iter>::type::template _Apply<_Iter>;
template <class _Tp>
  requires is_object_v<_Tp>
struct iterator_traits<_Tp> {
  typedef contiguous_iterator_tag iterator_concept;
};
template <class _Ip>
using iter_value_t = conditional_t<
    __is_primary_template<iterator_traits<remove_cvref_t<_Ip>>>::value,
    indirectly_readable_traits<remove_cvref_t<_Ip>>,
    iterator_traits<remove_cvref_t<_Ip>>>::value_type;
template <class _Tp> _Tp *addressof(_Tp &);
template <class _Bp, class _Dp>
constexpr bool is_base_of_v = __is_base_of(_Bp, _Dp);
template <class _Dp, class _Bp>
concept derived_from = is_base_of_v<_Bp, _Dp> && is_convertible_v<_Dp *, _Bp *>;
namespace ranges {
struct Trans_NS___iter_move___fn {
  template <class _Ip>
  auto operator()(_Ip __i) -> decltype(std::move(*__i));
};
auto iter_move = Trans_NS___iter_move___fn{};
} // namespace ranges
template <__dereferenceable _Tp>
  requires requires {
    { ranges::iter_move };
  }
using iter_rvalue_reference_t =
    decltype(ranges::iter_move(std::declval<_Tp>()));
template <class _In>
concept __indirectly_readable_impl =
    requires { typename iter_value_t<_In>; } &&
    common_reference_with<iter_reference_t<_In>, iter_value_t<_In> &> &&
    common_reference_with<iter_reference_t<_In>,
                          iter_rvalue_reference_t<_In>> &&
    common_reference_with<iter_rvalue_reference_t<_In>, iter_value_t<_In> &>;
template <class _In>
concept indirectly_readable = __indirectly_readable_impl<remove_cvref_t<_In>>;
template <class _Ip>
concept weakly_incrementable =
    !same_as<_Ip, bool> && movable<_Ip> && requires(_Ip __i) { __i; };
template <class _Ip>
concept input_or_output_iterator = requires(_Ip __i) {
  { __i };
} && weakly_incrementable<_Ip>;
template <class _Ip>
concept input_iterator =
    input_or_output_iterator<_Ip> && indirectly_readable<_Ip> && requires {
      typename _ITER_CONCEPT<_Ip>;
    } && derived_from<_ITER_CONCEPT<_Ip>, input_iterator_tag>;
namespace ranges {
struct __fn {
  template <class _Tp, decltype(sizeof(int)) _Np>
  constexpr auto operator()(_Tp (&__t)[_Np]) const
  {
    return __t;
  }
  template <class _Tp> constexpr auto operator()(_Tp __t) const {
    return static_cast<std::__decay_t<decltype((__t.begin()))>>(__t.begin());
  }
};
inline namespace {
auto begin = __fn{};
}
template <class _Tp>
using iterator_t = decltype(ranges::begin(std::declval<_Tp &>()));
inline namespace {
auto end = int{};
}
template <class _Derived>
class view_interface;
template <class _Op, class _Yp>
  requires is_convertible_v<_Op, view_interface<_Yp>>
void __is_derived_from_view_interface();
template <class _Tp>
bool enable_view = derived_from<_Tp, int> ||
                   requires { ranges::__is_derived_from_view_interface<_Tp, int>(); };
template <class>
concept range = requires { ranges::end; };
template <class _Tp>
concept input_range = range<_Tp> && input_iterator<iterator_t<_Tp>>;
template <class _Tp>
concept view = range<_Tp> && movable<_Tp> && enable_view<_Tp>;
template <class _Tp>
concept viewable_range =
    range<_Tp> &&
    ((view<remove_cvref_t<_Tp>> && constructible_from<remove_cvref_t<_Tp>>) ||
     (!view<remove_cvref_t<_Tp>> && (is_lvalue_reference_v<_Tp>)));
} // namespace ranges
template <decltype(sizeof(int))...> struct __tuple_indices;
template <class _IdxType, _IdxType... _Values> struct __integer_sequence {
  template <decltype(sizeof(int)) _Sp>
  using __to_tuple_indices = __tuple_indices<(_Values)...>;
};
template <decltype(sizeof(int)) _Ep, decltype(sizeof(int)) _Sp>
using __make_indices_imp =
    __make_integer_seq<__integer_sequence, decltype(sizeof(int)),
                       _Sp>::template __to_tuple_indices<_Sp>;
template <int _Ep, decltype(sizeof(int)) _Sp = 0> struct __make_tuple_indices {
  typedef __make_indices_imp<_Ep, _Sp> type;
};
template <class...> struct __tuple_types;
namespace ranges {
template <class _Derived>
class view_interface {};
} // namespace ranges
template <decltype(sizeof(int)) _Ip, class... _Types>
struct tuple_element<_Ip, __tuple_types<_Types...>> {
  using type = __type_pack_element<_Ip, _Types...>;
};
template <decltype(sizeof(int)) _Ip, class... _Tp>
struct tuple_element<_Ip, tuple<_Tp...>> {
  using type = tuple_element<_Ip, __tuple_types<_Tp...>>::type;
};

template <class... _Tp> struct tuple {
  int __value_;
  constexpr int get() { return __value_; }
};
template <int _Ip, class... _Tp> constexpr void get(tuple<_Tp...> __t) {
  __t.get();
}
namespace ranges {
template <class _Tp>
struct __range_adaptor_closure {};
template <range _Range>
  requires is_object_v<_Range>
struct ref_view : view_interface<ref_view<_Range>> {
  _Range *__range_;

  template <class _Tp>
  constexpr ref_view(_Tp &&__t)
      : __range_(std::addressof(static_cast<_Range &>(__t))) {}
  constexpr iterator_t<_Range> begin() { return ranges::begin(*__range_); }
};
template <class _Range> ref_view(_Range &) -> ref_view<_Range>;
} // namespace ranges
namespace ranges::views {
struct __fn : __range_adaptor_closure<__fn> {
  template <class _Tp> auto operator()(_Tp &&__t) const {
    return ranges::ref_view{__t};
  }
};
inline namespace {
auto all = __fn{};
}
template <ranges::viewable_range _Range>
using all_t = decltype(views::all(std::declval<_Range>()));
} // namespace ranges::views
namespace ranges {
template <input_range _View, decltype(sizeof(int)) _Np>
struct elements_view : view_interface<elements_view<_View, _Np>> {
  class __iterator;

  constexpr elements_view(_View __base) : __base_(std::move(__base)) {}
  constexpr auto begin() const { return __iterator(ranges::begin(__base_)); }

  _View __base_ = _View();
};
template <input_range _View, decltype(sizeof(int)) _Np>
struct elements_view<_View, _Np>::__iterator {
  iterator_t<_View> __current_;

  constexpr void operator*() {
    auto a = *__current_;
  }
};
namespace views {
namespace __elements {
template <int _Np> struct __fn : __range_adaptor_closure<__fn<_Np>> {
  template <class _Range>
  constexpr auto operator()(_Range &&__range) const
      -> decltype(elements_view<all_t<_Range>, _Np>(__range)) {
    return elements_view<all_t<_Range>, _Np>(__range);
  }
};
} // namespace __elements
inline namespace {
template <decltype(sizeof(int)) _Np> auto elements = __elements::__fn<_Np>{};
}
} // namespace views
} // namespace ranges
} // namespace
} // namespace std
constexpr bool test() {
  std::tuple<short, long> ts[]{{}};


  auto ev = std::ranges::views::elements<1>(ts);
  auto it = ev.begin();

  *it;
  return true;
}
static_assert(test());
