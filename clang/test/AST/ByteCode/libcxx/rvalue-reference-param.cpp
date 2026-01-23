// RUN: %clang_cc1 -std=c++2c -verify=expected,both %s -fexperimental-new-constant-interpreter
// RUN: %clang_cc1 -std=c++2c -verify=ref,both      %s

template <int __v> struct integral_constant {
  static const int value = __v;
};
template <bool _Val> using _BoolConstant = integral_constant<_Val>;
template <class _From, class _To>
constexpr bool is_convertible_v = __is_convertible(_From, _To);
template <class _Tp> _Tp __declval(int);
template <class _Tp> decltype(__declval<_Tp>(0)) declval();
template <class _Tp, class _Up>
using _IsSame = _BoolConstant<__is_same(_Tp, _Up)>;
template <class _If> struct conditional {
  using type = _If;
};
template <bool, class _IfRes, class>
using conditional_t = conditional<_IfRes>::type;
template <class _Tp, class>
concept __weakly_equality_comparable_with = requires(_Tp __t) { __t; };
template <bool, class _Tp = void> using __enable_if_t = _Tp;
template <template <class> class, class>
integral_constant<true> __sfinae_test_impl(int);
template <template <class> class _Templ, class... _Args>
using _IsValidExpansion = decltype(__sfinae_test_impl<_Templ, _Args...>(0));
template <class _Tp>
using __test_for_primary_template =
    __enable_if_t<_IsSame<_Tp, typename _Tp::__primary_template>::value>;
template <class _Tp>
using __is_primary_template =
    _IsValidExpansion<__test_for_primary_template, _Tp>;
template <class _Ip>
using iter_difference_t =
    conditional_t<__is_primary_template<_Ip>::value, _Ip, _Ip>::difference_type;
template <int> struct _OrImpl {
  template <class, class _First>
  using _Result = _OrImpl<!_First::value>::template _Result<_First>;
};
template <> struct _OrImpl<false> {
  template <class _Res> using _Result = _Res;
};
template <class... _Args>
using _Or = _OrImpl<sizeof...(_Args)>::template _Result<_Args...>;
struct input_iterator_tag {};
template <class _Dp, class _Bp>
concept derived_from = is_convertible_v<_Dp, _Bp>;
template <class _Ip>
concept input_or_output_iterator = requires(_Ip __i) { __i; };
template <class _Sp, class _Ip>
concept sentinel_for = __weakly_equality_comparable_with<_Sp, _Ip>;
struct __iter_concept_category_test {
  template <class> using _Apply = input_iterator_tag;
};
struct __test_iter_concept
    : _IsValidExpansion<__iter_concept_category_test::_Apply, int>,
      __iter_concept_category_test {};
struct __iter_concept_cache {
  using type = _Or<int, __test_iter_concept>;
};
template <class _Iter>
using _ITER_CONCEPT = __iter_concept_cache::type::_Apply<_Iter>;
template <class _Ip>
concept input_iterator = derived_from<_ITER_CONCEPT<_Ip>, input_iterator_tag>;
template <class _T1, class _T2> struct pair {
  _T1 first;
  _T2 second;
};
struct {
  template <class _Tp> auto operator()(_Tp __t) { return __t.begin(); }
} begin;
template <class _Tp> using iterator_t = decltype(begin(declval<_Tp>()));
template <class _Tp>
concept __member_size = requires(_Tp __t) { __t; };
struct {
  template <__member_size _Tp> constexpr void operator()(_Tp &&__t) {
    __t.size(); // both-note 2{{in instantiation}}
  }
} size;
template <class _Tp>
concept range = requires(_Tp __t) { __t; };
template <class _Tp>
concept input_range = input_iterator<_Tp>;
template <range _Rp>
using range_difference_t = iter_difference_t<iterator_t<_Rp>>;
struct {
  template <range _Rp> constexpr range_difference_t<_Rp> operator()(_Rp &&__r) {
    size(__r); // both-note 2{{in instantiation}}
  } // both-warning {{does not return a value}}
} distance;
template <input_or_output_iterator _Iter, sentinel_for<_Iter> _Sent>
struct subrange {
  _Iter __begin_;
  _Sent __end_;
  _Iter begin();
  constexpr _Iter size() { __end_ - __begin_; } // both-warning {{does not return a value}} \
                                                // both-note {{in instantiation}}
};
struct {
  template <input_range _Range1, input_range _Range2>
  void operator()(_Range1 &&__range1, _Range2) {
    (void)(distance(__range1) != 0); // both-note 3{{in instantiation}}
  }
} equal;
template <class _Owner> struct __key_value_iterator {
  using difference_type = _Owner::difference_type;
  constexpr friend difference_type operator-(__key_value_iterator,
                                             __key_value_iterator &) {} // both-warning {{does not return a value}}
};
struct flat_multimap {
  template <bool> using __iterator = __key_value_iterator<flat_multimap>;
  using difference_type =
      decltype(static_cast<int *>(nullptr) - static_cast<int *>(nullptr));
  pair<__iterator<true>, __iterator<true>> equal_range(const char *);
} test_expected_range;
void test() {
  flat_multimap m;
  auto test_found = [](auto map, auto expected_key, int) {
    auto [first, last] = map.equal_range(expected_key);
    equal(subrange(first, last), test_expected_range); // both-note 3{{in instantiation}}
  };
  test_found(m, "", {});
}
