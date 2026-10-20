// RUN: %clang_cc1 -std=c++2c -fexperimental-new-constant-interpreter -verify=expected,both %s
// RUN: %clang_cc1 -std=c++2c  -verify=ref,both %s

/// This used to cause an assertion failure because we were deallocating a
/// dynamic block that was already dead.

namespace std {
inline namespace __1 {
template <class _Tp, _Tp __v> struct integral_constant {
  static inline constexpr const _Tp value = __v;
};
typedef integral_constant<bool, true> true_type;
typedef integral_constant<bool, false> false_type;
template <bool _Val> using _BoolConstant = integral_constant<bool, _Val>;
template <class _Tp> using __remove_cv_t = __remove_cv(_Tp);
template <class _Tp> using remove_cv_t = __remove_cv_t<_Tp>;
template <class _Tp>
inline constexpr bool is_lvalue_reference_v = __is_lvalue_reference(_Tp);
template <class _Tp>
using __libcpp_remove_reference_t = __remove_reference_t(_Tp);
template <class _Tp>
constexpr _Tp &&forward(__libcpp_remove_reference_t<_Tp> &__t) noexcept;
template <bool _Bp, class _If, class _Then> struct conditional {
  using type = _If;
};
template <bool _Bp, class _IfRes, class _ElseRes>
using conditional_t = typename conditional<_Bp, _IfRes, _ElseRes>::type;
template <class _Tp, class _Up>
using _IsSame = _BoolConstant<__is_same(_Tp, _Up)>;
using size_t = decltype(sizeof(int));
template <class _Tp> using __decay_t = __decay(_Tp);
template <class _Tp> using decay_t = __decay_t<_Tp>;
template <bool, class _Tp = void> struct enable_if;
template <class _Tp> struct enable_if<true, _Tp> {
  typedef _Tp type;
};
template <bool _Bp, class _Tp = void>
using __enable_if_t = typename enable_if<_Bp, _Tp>::type;
template <class _Bp, class _Dp>
inline constexpr bool is_base_of_v = __is_base_of(_Bp, _Dp);
template <class _Tp> _Tp &&__declval(int);
template <class _Tp> decltype(std::__declval<_Tp>(0)) declval() noexcept;
template <class _Ret, class _Fp, class... _Args> struct __invokable_r {};
template <class _Fp, class... _Args>
using __is_invocable = __invokable_r<void, _Fp, _Args...>;
template <class _Func, class... _Args>
inline const bool __is_invocable_v = __is_invocable<_Func, _Args...>::value;
template <class _Func, class... _Args>
struct __invoke_result
    : enable_if<__is_invocable_v<_Func, _Args...>,
                typename __invokable_r<void, _Func, _Args...>::_Result> {};
template <class _Fn, class... _Args>
struct invoke_result : __invoke_result<_Fn, _Args...> {};
template <class _Tp, class... _Args>
inline constexpr bool is_constructible_v = __is_constructible(_Tp, _Args...);
template <class _Tp>
constexpr __libcpp_remove_reference_t<_Tp> &&move(_Tp &&__t) noexcept;
template <class _Tp> inline constexpr bool is_enum_v = __is_enum(_Tp);
template <class _From, class _To>
inline constexpr bool is_convertible_v = __is_convertible(_From, _To);
template <class _From, class _To>
concept convertible_to = is_convertible_v<_From, _To> &&
                         requires { static_cast<_To>(std::declval<_From>()); };
template <class _Tp, class _Up>
concept __same_as_impl = _IsSame<_Tp, _Up>::value;
template <class _Tp, class _Up>
concept same_as = __same_as_impl<_Tp, _Up> && __same_as_impl<_Up, _Tp>;
template <class _Tp> using __remove_cvref_t = __remove_cvref(_Tp);
template <class _Tp> using remove_cvref_t = __remove_cvref_t<_Tp>;
template <class _Tp>
using __make_const_lvalue_ref = const __libcpp_remove_reference_t<_Tp> &;
template <class _Lhs, class _Rhs>
concept assignable_from =
    is_lvalue_reference_v<_Lhs> &&
    requires(_Lhs __lhs, _Rhs &&__rhs) {
      { __lhs = std::forward<_Rhs>(__rhs) } -> same_as<_Lhs>;
    };
template <class _Tp>
struct is_nothrow_destructible
    : integral_constant<bool, __is_nothrow_destructible(_Tp)> {};
template <class _Tp>
inline constexpr bool is_nothrow_destructible_v =
    is_nothrow_destructible<_Tp>::value;
template <class _Tp>
concept destructible = is_nothrow_destructible_v<_Tp>;
template <class _Tp, class... _Args>
concept constructible_from =
    destructible<_Tp> && is_constructible_v<_Tp, _Args...>;
template <class _Tp>
concept __default_initializable = requires { ::new _Tp; };
template <class _Tp>
concept default_initializable = constructible_from<_Tp> && requires {
  _Tp{};
} && __default_initializable<_Tp>;
template <class _Tp>
concept move_constructible =
    constructible_from<_Tp, _Tp> && convertible_to<_Tp, _Tp>;
template <class _Tp> inline constexpr bool is_class_v = __is_class(_Tp);
template <class _Tp> inline constexpr bool is_union_v = __is_union(_Tp);
template <class _Tp>
concept __class_or_enum = is_class_v<_Tp> || is_union_v<_Tp> || is_enum_v<_Tp>;
namespace ranges {
namespace __swap {
template <class _Tp, class _Up>
concept __unqualified_swappable_with =
    (__class_or_enum<remove_cvref_t<_Tp>> ||
     __class_or_enum<remove_cvref_t<_Up>>) &&
    requires(_Tp &&__t, _Up &&__u) {
      swap(std::forward<_Tp>(__t), std::forward<_Up>(__u));
    };
template <class _Tp>
concept __exchangeable = !__unqualified_swappable_with<_Tp &, _Tp &> &&
                         move_constructible<_Tp> && assignable_from<_Tp &, _Tp>;
struct __fn {
  template <__exchangeable _Tp>
  constexpr void operator()(_Tp &__x, _Tp &__y) const;
};
} // namespace __swap
inline namespace __cpo {
inline constexpr auto swap = __swap::__fn{};
}
} // namespace ranges
template <class _Tp> inline constexpr bool is_object_v = __is_object(_Tp);
template <class _Tp>
concept movable = is_object_v<_Tp> && move_constructible<_Tp> &&
                  assignable_from<_Tp &, _Tp>;
template <class _Tp>
concept copyable =
    assignable_from<_Tp &, const _Tp &> && assignable_from<_Tp &, const _Tp>;
template <class _Dp, class _Bp>
concept derived_from =
    is_base_of_v<_Bp, _Dp> &&
    is_convertible_v<const volatile _Dp *, const volatile _Bp *>;
template <class _Tp>
concept __boolean_testable_impl = convertible_to<_Tp, bool>;
template <class _Tp>
concept __boolean_testable =
    __boolean_testable_impl<_Tp> && requires(_Tp &&__t) {
      { !std::forward<_Tp>(__t) } -> __boolean_testable_impl;
    };
template <class _Tp, class _Up>
concept __weakly_equality_comparable_with = requires(
    __make_const_lvalue_ref<_Tp> __t, __make_const_lvalue_ref<_Up> __u) {
  { __u != __t } -> __boolean_testable;
};
template <class _Tp>
concept semiregular = copyable<_Tp> && default_initializable<_Tp>;
template <template <class...> class _Templ, class... _Args,
          class = _Templ<_Args...>>
true_type __sfinae_test_impl(int);
template <template <class...> class, class...>
false_type __sfinae_test_impl(...);
template <template <class...> class _Templ, class... _Args>
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
template <class _Tp>
concept __has_member_value_type = requires { typename _Tp::value_type; };
template <class> struct indirectly_readable_traits;
template <class _Tp>
struct indirectly_readable_traits<_Tp *> : __cond_value_type<_Tp> {};
template <__has_member_value_type _Tp>
struct indirectly_readable_traits<_Tp>
    : __cond_value_type<typename _Tp::value_type> {};
template <bool> struct _OrImpl;
template <> struct _OrImpl<true> {
  template <class _Res, class _First, class... _Rest>
  using _Result =
      typename _OrImpl<!bool(_First::value) && sizeof...(_Rest) != 0>::
          template _Result<_First, _Rest...>;
};
template <> struct _OrImpl<false> {
  template <class _Res, class...> using _Result = _Res;
};
template <class... _Args>
using _Or = typename _OrImpl<sizeof...(_Args) !=
                             0>::template _Result<false_type, _Args...>;
template <class _Tp> using __with_reference = _Tp &;
template <class _Tp>
concept __can_reference = requires { typename __with_reference<_Tp>; };
template <class _Tp>
concept __dereferenceable = requires(_Tp &__t) {
  { *__t } -> __can_reference;
};
template <__dereferenceable _Tp>
using iter_reference_t = decltype(*std::declval<_Tp &>());
struct input_iterator_tag {};
struct forward_iterator_tag : public input_iterator_tag {};
struct bidirectional_iterator_tag : public forward_iterator_tag {};
struct random_access_iterator_tag : public bidirectional_iterator_tag {};
template <class _Iter> struct __iter_traits_cache;
template <class _Iter>
using _ITER_TRAITS = typename __iter_traits_cache<_Iter>::type;
struct __iter_concept_concept_test {
  template <class _Iter>
  using _Apply = typename _ITER_TRAITS<_Iter>::iterator_concept;
};
struct __iter_concept_category_test {
  template <class _Iter>
  using _Apply = typename _ITER_TRAITS<_Iter>::iterator_category;
};
struct __iter_concept_random_fallback {
  template <class _Iter>
  using _Apply =
      __enable_if_t<__is_primary_template<iterator_traits<_Iter>>::value,
                    random_access_iterator_tag>;
};
template <class _Iter, class _Tester>
struct __test_iter_concept : _IsValidExpansion<_Tester::template _Apply, _Iter>,
                             _Tester {};
template <class _Iter> struct __iter_concept_cache {
  using type = _Or<__test_iter_concept<_Iter, __iter_concept_concept_test>,
                   __test_iter_concept<_Iter, __iter_concept_category_test>,
                   __test_iter_concept<_Iter, __iter_concept_random_fallback>>;
};
template <class _Iter>
using _ITER_CONCEPT =
    typename __iter_concept_cache<_Iter>::type::template _Apply<_Iter>;
template <class _Ip> struct iterator_traits {
  using __primary_template = iterator_traits;
};
template <class _Ip>
using iter_value_t = typename conditional_t<
    __is_primary_template<iterator_traits<remove_cvref_t<_Ip>>>::value,
    indirectly_readable_traits<remove_cvref_t<_Ip>>,
    iterator_traits<remove_cvref_t<_Ip>>>::value_type;
namespace ranges {
namespace __iter_move {
template <class _Tp>
concept __move_deref = requires(_Tp &&__t) {
  requires is_lvalue_reference_v<decltype(*__t)>;
};
struct __fn {
  template <class _Ip>
    requires __move_deref<_Ip>
  constexpr auto operator()(_Ip &&__i) const
      -> decltype(std::move(*std::forward<_Ip>(__i))) {
    return std::move(*std::forward<_Ip>(__i));
  }
  template <class _Ip>
  constexpr auto operator()(_Ip &&__i) const
      -> decltype(*std::forward<_Ip>(__i));
};
} // namespace __iter_move
inline namespace __cpo {
inline constexpr auto iter_move = __iter_move::__fn{};
}
} // namespace ranges
template <__dereferenceable _Tp>
  requires requires(_Tp &__t) {
    { ranges::iter_move(__t) } -> __can_reference;
  }
using iter_rvalue_reference_t =
    decltype(ranges::iter_move(std::declval<_Tp &>()));
template <class _Ptr, class = void> struct __pointer_traits_impl {};
template <class _Tp> inline constexpr bool is_pointer_v = __is_pointer(_Tp);
template <class _Ip>
concept input_or_output_iterator = requires(_Ip __i) {
  { *__i } -> __can_reference;
};
template <class _Sp, class _Ip>
concept sentinel_for = semiregular<_Sp> && input_or_output_iterator<_Ip> &&
                       __weakly_equality_comparable_with<_Sp, _Ip>;
template <class _Ip>
concept input_iterator =
    input_or_output_iterator<_Ip> && requires {
      typename _ITER_CONCEPT<_Ip>;
    } && derived_from<_ITER_CONCEPT<_Ip>, input_iterator_tag>;
namespace ranges {
namespace __begin {
struct __fn {
  template <class _Tp> constexpr auto operator()(_Tp &&__t) const {
    return static_cast<::std::__decay_t<decltype((__t.begin()))>>(__t.begin());
  }
};
} // namespace __begin
inline namespace __cpo {
inline constexpr auto begin = __begin::__fn{};
}
template <class _Tp>
using iterator_t = decltype(ranges::begin(std::declval<_Tp &>()));
namespace __end {
struct __fn {
  template <class _Tp> constexpr auto operator()(_Tp &&__t) const {}
};
} // namespace __end
inline namespace __cpo {
inline constexpr auto end = __end::__fn{};
}
} // namespace ranges
namespace ranges {
template <class _Tp>
concept range = requires(_Tp &__t) { ranges::end(__t); };
template <class _Tp>
concept input_range = range<_Tp> && input_iterator<iterator_t<_Tp>>;
template <range _Rp> using range_value_t = iter_value_t<iterator_t<_Rp>>;
} // namespace ranges
} // namespace __1
} // namespace std
constexpr void *operator new(std::size_t, void *__p) noexcept; // both-warning {{not defined}}
namespace std {
inline namespace __1 {
template <class _Tp, class... _Args,
          class = decltype(::new (std::declval<void *>())
                               _Tp(std::declval<_Args>()...))>
constexpr _Tp *construct_at(_Tp *__location, _Args &&...__args) {
  return ::new (static_cast<void *>(__location)) _Tp(std::forward<_Args>(__args)...); // both-note {{here}}
}
enum class __element_count : size_t;
template <class _Tp, class _Up>
constexpr _Tp *__constexpr_memmove(_Tp *__dest, _Up *__src,
                                   __element_count __n) {
  size_t __count = static_cast<size_t>(__n);
  ::__builtin_memcpy(__dest, __src, __count * sizeof(_Tp));
  return __dest;
};
namespace ranges {
template <class _From, class _To>
concept __convertible_to_non_slicing =
    convertible_to<_From, _To>;
template <input_or_output_iterator _Iter, sentinel_for<_Iter> _Sent = _Iter>
class subrange  {
private:
  _Iter __begin_ = _Iter();
  _Sent __end_ = _Sent();

public:
  constexpr subrange(__convertible_to_non_slicing<_Iter> auto __iter,
                     _Sent __sent)
      : __begin_(std::move(__iter)), __end_(std::move(__sent)) {}
  constexpr _Iter begin() { return std::move(__begin_); }
  constexpr _Sent end() const;
};
template <input_or_output_iterator _Iter, sentinel_for<_Iter> _Sent>
subrange(_Iter, _Sent) -> subrange<_Iter, _Sent>;
} // namespace ranges
template <class _Tp> class allocator {
public:
  constexpr _Tp *allocate(size_t __n) {
    return static_cast<_Tp *>(::operator new(__n * sizeof(_Tp)));
  }
  constexpr void deallocate(_Tp *__p, size_t __n) noexcept {
    ::operator delete(__p);
  }
};
template <class _CharT> struct char_traits;
template <class _CharT, class _Traits = char_traits<_CharT>,
          class _Allocator = allocator<_CharT>>
class basic_string;
using string = basic_string<char>;
template <class _Iter>
class move_iterator {
private:
public:
  using iterator_type = _Iter;
  using value_type = iter_value_t<_Iter>;
  using reference = iter_rvalue_reference_t<_Iter>;
  constexpr explicit move_iterator(_Iter __i) : __current_(std::move(__i)) {}
  constexpr move_iterator();
  constexpr reference operator*() const {
    return ranges::iter_move(__current_);
  }
  _Iter __current_;
};
namespace ranges {
template <class _T1> struct min_max_result {
  _T1 min;
};
template <class _T1> using minmax_result = min_max_result<_T1>;
struct __minmax {
  template <input_range _Range>
  constexpr ranges::minmax_result<range_value_t<_Range>>
  operator()(_Range &&__r) const {
    auto __first = ranges::begin(__r);
    using _ValueT = range_value_t<_Range>;
    ranges::minmax_result<_ValueT> __result = {*__first};
    return __result;
  }
};
inline namespace __cpo {
inline constexpr auto minmax = __minmax{};
}
} // namespace ranges
template <> struct char_traits<char> {
  using char_type = char;
  static constexpr int compare(const char_type *__lhs, const char_type *__rhs,
                               size_t __count) noexcept {
    return __builtin_memcmp(__lhs, __rhs, __count);
  }
  static inline size_t constexpr length(const char_type *__s) noexcept {
    return __builtin_strlen(__s);
  }
  static inline constexpr char_type *
  copy(char_type *__s1, const char_type *__s2, size_t __n) noexcept {
    std::__constexpr_memmove(__s1, __s2, __element_count(__n));
    return __s1;
  }
};
template <class _CharT, class _Traits, class _Allocator> class basic_string {
  using traits_type = _Traits;
  using value_type = _CharT;
  using allocator_type = _Allocator;

private:
  struct __long {
    struct {
      unsigned __is_long_ : 1;
      unsigned __cap_ : sizeof(unsigned) * 8 - 1;
    };
    unsigned __size_;
    char *__data_;
  };
  struct __short {
    struct {
      unsigned char __is_long_ : 1;
    };
  };
  union __rep {
    __short __s;
    __long __l;
  };
  __rep __rep_;
  allocator_type __alloc_;

public:
  constexpr basic_string(const basic_string &__str) : __alloc_(__str.__alloc_) {
    __init_copy_ctor_external(__str.__rep_.__l.__data_,
                              __str.__rep_.__l.__size_);
  }
  constexpr basic_string(basic_string &&__str)
      : __rep_(__str.__rep_), __alloc_(std::move(__str.__alloc_)) {
    __str.__rep_ = __rep();
  }
  constexpr basic_string(const _CharT *__s) {
    __init_copy_ctor_external(__s, traits_type::length(__s));
  }
  inline constexpr ~basic_string() {
    if (__is_long())
      __alloc_.deallocate(__rep_.__l.__data_, __rep_.__l.__cap_*2);
  }
  constexpr unsigned size() const noexcept { return __rep_.__l.__size_; }
  constexpr const value_type *data() const noexcept { return __rep_.__l.__data_; }

private:
  constexpr bool __is_long() const {
    if (__builtin_constant_p(__rep_.__l.__is_long_)) {
      return __rep_.__l.__is_long_;
    }
    return __rep_.__s.__is_long_;
  }
  static constexpr void __begin_lifetime(char *__begin, unsigned __n) {
    for (unsigned __i = 0; __i != __n; ++__i)
      std::construct_at(&__begin[__i]);
  }
  constexpr void __init_copy_ctor_external(const value_type *__s,
                                           unsigned __sz);
};
template <class _CharT, class _Traits, class _Allocator>
constexpr void
basic_string<_CharT, _Traits, _Allocator>::__init_copy_ctor_external(
    const value_type *__s, unsigned __sz) {
  char *__p;
  auto __allocation = __alloc_.allocate(__sz + 1);
  __p = __allocation;
  __begin_lifetime(__p, __sz + 1);
  __rep_.__l.__data_ = __p;
  __rep_.__l.__cap_ = (__sz + 1) / 2;
  __rep_.__l.__is_long_ = true;
  __rep_.__l.__size_ = __sz;
  traits_type::copy(__p, __s, __sz + 1);
}
template <class _CharT, class _Traits, class _Allocator>
inline constexpr bool
operator==(const basic_string<_CharT, _Traits, _Allocator> &__lhs,
           const basic_string<_CharT, _Traits, _Allocator> &__rhs) noexcept {
  return _Traits::compare(__lhs.data(), __rhs.data(), __lhs.size()) == 0;
}
} // namespace __1
} // namespace std
template <class It> class cpp20_input_iterator {
  It it_;

public:
  using value_type = std::iter_value_t<It>;
  constexpr explicit cpp20_input_iterator(It it) : it_(it) {}
  constexpr decltype(auto) operator*() const { return *it_; }
  constexpr void operator++(int);
  friend constexpr It base(const cpp20_input_iterator &i) { return i.it_; }
};
template <class It> class sentinel_wrapper {
public:
  explicit sentinel_wrapper() = default;
  constexpr explicit sentinel_wrapper(const It &it) : base_(base(it)) {}
  constexpr bool operator==(const It &other) const;

private:
  decltype(base(std::declval<It>())) base_;
};
constexpr bool test_range() {
  const std::string str{
      "this long string will be dynamically "
      "allocatedasfajkshdfasdhjkfahjksdgfjkasdhkfgaksdgfhkasghjkdfgjhasdghjfgah"
      "jksdgjfhkasghjkdfghjasgjdfghasdghjkfgajksdgfgajsdfghjkgashjkdfghjasjdhkf"
      "gahjsdfghjkgashjdghjfkasghjkdfasgjkdhfgajkshdfgkjashdgfkjasghdfkjghasdkf"
      "hgaskdjghfasdfasdfasdfasdfasdf"};
  std::string a[] = {str};
  auto range = std::ranges::subrange(
      cpp20_input_iterator(std::move_iterator(a)),
      sentinel_wrapper(cpp20_input_iterator(std::move_iterator(a + 1))));
  auto ret = std::ranges::minmax(range);
  if (ret.min != str)
    __builtin_abort();
  return true;
}
static_assert(test_range());
