// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// std::bad_optional_access
// optional base classes

#ifndef _LIBCPP_OPTIONAL_COMMON_H
#define _LIBCPP_OPTIONAL_COMMON_H

#include <__assert>
#include <__config>
#include <__exception/exception.h>
#include <__functional/invoke.h>
#include <__fwd/format.h>
#include <__fwd/optional.h>
#include <__iterator/bounded_iter.h>
#include <__iterator/capacity_aware_iterator.h>
#include <__memory/addressof.h>
#include <__memory/construct_at.h>
#include <__ranges/enable_borrowed_range.h>
#include <__ranges/enable_view.h>
#include <__tuple/sfinae_helpers.h>
#include <__type_traits/add_pointer.h>
#include <__type_traits/enable_if.h>
#include <__type_traits/integral_constant.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_array.h>
#include <__type_traits/is_assignable.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_convertible.h>
#include <__type_traits/is_destructible.h>
#include <__type_traits/is_nothrow_assignable.h>
#include <__type_traits/is_nothrow_constructible.h>
#include <__type_traits/is_object.h>
#include <__type_traits/is_reference.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_swappable.h>
#include <__type_traits/is_trivially_assignable.h>
#include <__type_traits/is_trivially_constructible.h>
#include <__type_traits/is_trivially_destructible.h>
#include <__type_traits/reference_constructs_from_temporary.h>
#include <__type_traits/remove_const.h>
#include <__type_traits/remove_cv.h>
#include <__type_traits/remove_reference.h>
#include <__utility/forward.h>
#include <__utility/in_place.h>
#include <__utility/move.h>
#include <__utility/swap.h>
#include <__verbose_abort>
#include <initializer_list>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

namespace std // purposefully not using versioning namespace
{

class _LIBCPP_EXPORTED_FROM_ABI bad_optional_access : public exception {
public:
  bad_optional_access() _NOEXCEPT                                      = default;
  bad_optional_access(const bad_optional_access&) _NOEXCEPT            = default;
  bad_optional_access& operator=(const bad_optional_access&) _NOEXCEPT = default;
  // Get the key function ~bad_optional_access() into the dylib
  ~bad_optional_access() _NOEXCEPT override;
  [[__nodiscard__]] const char* what() const _NOEXCEPT override;
};
} // namespace std

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

[[noreturn]] inline void __throw_bad_optional_access() {
#  if _LIBCPP_HAS_EXCEPTIONS
  throw bad_optional_access();
#  else
  _LIBCPP_VERBOSE_ABORT("bad_optional_access was thrown in -fno-exceptions mode");
#  endif
}

struct __optional_construct_from_invoke_tag {};

template <class _Tp, bool = is_trivially_destructible<_Tp>::value>
struct __optional_destruct_base;

template <class _Tp>
struct __optional_destruct_base<_Tp, false> {
  typedef _Tp value_type;
  static_assert(is_object_v<value_type>, "instantiation of optional with a non-object type is undefined behavior");
  union {
    char __null_state_;
    remove_cv_t<value_type> __val_;
  };
  bool __engaged_;

  _LIBCPP_CONSTEXPR_SINCE_CXX20 ~__optional_destruct_base() {
    if (__engaged_)
      __val_.~value_type();
  }

  constexpr __optional_destruct_base() noexcept : __null_state_(), __engaged_(false) {}

  template <class... _Args>
  constexpr explicit __optional_destruct_base(in_place_t, _Args&&... __args)
      : __val_(std::forward<_Args>(__args)...), __engaged_(true) {}

#  if _LIBCPP_STD_VER >= 23
  template <class _Fp, class... _Args>
  constexpr explicit __optional_destruct_base(__optional_construct_from_invoke_tag, _Fp&& __f, _Args&&... __args)
      : __val_(std::invoke(std::forward<_Fp>(__f), std::forward<_Args>(__args)...)), __engaged_(true) {}
#  endif

  _LIBCPP_CONSTEXPR_SINCE_CXX20 void reset() noexcept {
    if (__engaged_) {
      __val_.~value_type();
      __engaged_ = false;
    }
  }
};

template <class _Tp>
struct __optional_destruct_base<_Tp, true> {
  typedef _Tp value_type;
  static_assert(is_object_v<value_type>, "instantiation of optional with a non-object type is undefined behavior");
  union {
    char __null_state_;
    remove_cv_t<value_type> __val_;
  };
  bool __engaged_;

  constexpr __optional_destruct_base() noexcept : __null_state_(), __engaged_(false) {}

  template <class... _Args>
  constexpr explicit __optional_destruct_base(in_place_t, _Args&&... __args)
      : __val_(std::forward<_Args>(__args)...), __engaged_(true) {}

#  if _LIBCPP_STD_VER >= 23
  template <class _Fp, class... _Args>
  constexpr __optional_destruct_base(__optional_construct_from_invoke_tag, _Fp&& __f, _Args&&... __args)
      : __val_(std::invoke(std::forward<_Fp>(__f), std::forward<_Args>(__args)...)), __engaged_(true) {}
#  endif

  _LIBCPP_CONSTEXPR_SINCE_CXX20 void reset() noexcept {
    if (__engaged_) {
      __engaged_ = false;
    }
  }
};

template <class _Tp, bool = is_reference<_Tp>::value>
struct __optional_storage_base : __optional_destruct_base<_Tp> {
  using __base _LIBCPP_NODEBUG = __optional_destruct_base<_Tp>;
  using value_type             = _Tp;
  using __base::__base;

  [[nodiscard]] constexpr bool has_value() const noexcept { return this->__engaged_; }

  constexpr value_type& __get() & noexcept { return this->__val_; }
  constexpr const value_type& __get() const& noexcept { return this->__val_; }
  constexpr value_type&& __get() && noexcept { return std::move(this->__val_); }
  constexpr const value_type&& __get() const&& noexcept { return std::move(this->__val_); }

  template <class... _Args>
  _LIBCPP_CONSTEXPR_SINCE_CXX20 void __construct(_Args&&... __args) {
    _LIBCPP_ASSERT_INTERNAL(!has_value(), "__construct called for engaged __optional_storage");
    std::__construct_at(std::addressof(this->__val_), std::forward<_Args>(__args)...);
    this->__engaged_ = true;
  }

  template <class _That>
  _LIBCPP_CONSTEXPR_SINCE_CXX20 void __construct_from(_That&& __opt) {
    if (__opt.has_value())
      __construct(std::forward<_That>(__opt).__get());
  }

  template <class _That>
  _LIBCPP_CONSTEXPR_SINCE_CXX20 void __assign_from(_That&& __opt) {
    if (this->__engaged_ == __opt.has_value()) {
      if (this->__engaged_)
        static_cast<_Tp&>(this->__val_) = std::forward<_That>(__opt).__get();
    } else {
      if (this->__engaged_)
        this->reset();
      else
        __construct(std::forward<_That>(__opt).__get());
    }
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX20 void
  __swap(__optional_storage_base& __rhs) noexcept(is_nothrow_move_constructible_v<_Tp> && is_nothrow_swappable_v<_Tp>) {
    using std::swap;
    if (this->has_value() == __rhs.has_value()) {
      if (this->has_value())
        swap(this->__get(), __rhs.__get());
    } else {
      if (this->has_value()) {
        __rhs.__construct(std::move(this->__get()));
        this->reset();
      } else {
        this->__construct(std::move(__rhs.__get()));
        __rhs.reset();
      }
    }
  }

  // [optional.observe]
  constexpr add_pointer_t<_Tp const> operator->() const noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(this->has_value(), "optional operator-> called on a disengaged value");
    return std::addressof(this->__get());
  }

  constexpr add_pointer_t<_Tp> operator->() noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(this->has_value(), "optional operator-> called on a disengaged value");
    return std::addressof(this->__get());
  }

  [[nodiscard]] constexpr const _Tp& operator*() const& noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(this->has_value(), "optional operator* called on a disengaged value");
    return this->__get();
  }

  [[nodiscard]] constexpr _Tp& operator*() & noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(this->has_value(), "optional operator* called on a disengaged value");
    return this->__get();
  }

  [[nodiscard]] constexpr _Tp&& operator*() && noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(this->has_value(), "optional operator* called on a disengaged value");
    return std::move(this->__get());
  }

  [[nodiscard]] constexpr const _Tp&& operator*() const&& noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(this->has_value(), "optional operator* called on a disengaged value");
    return std::move(this->__get());
  }

  [[nodiscard]] constexpr _Tp const& value() const& {
    if (!this->has_value())
      std::__throw_bad_optional_access();
    return this->__get();
  }

  [[nodiscard]] constexpr _Tp& value() & {
    if (!this->has_value())
      std::__throw_bad_optional_access();
    return this->__get();
  }

  [[nodiscard]] constexpr _Tp&& value() && {
    if (!this->has_value())
      std::__throw_bad_optional_access();
    return std::move(this->__get());
  }

  [[nodiscard]] constexpr _Tp const&& value() const&& {
    if (!this->has_value())
      std::__throw_bad_optional_access();
    return std::move(this->__get());
  }
};

template <class _Tp>
struct __optional_storage_base<_Tp, true> {
  using value_type                 = _Tp;
  using __raw_type _LIBCPP_NODEBUG = remove_reference_t<_Tp>;
  __raw_type* __value_;

  constexpr __optional_storage_base() noexcept : __value_(nullptr) {}

  template <class _Up>
  constexpr void __convert_init_ref_val(_Up&& __val) noexcept {
    _Tp& __r(std::forward<_Up>(__val));
    __value_ = std::addressof(__r);
  }

  template <class _UArg>
  constexpr explicit __optional_storage_base(in_place_t, _UArg&& __uarg) {
    static_assert(!__reference_constructs_from_temporary_v<_Tp, _UArg>,
                  "Attempted to construct a reference element in optional from a "
                  "possible temporary");
    __convert_init_ref_val(std::forward<_UArg>(__uarg));
  }

#  if _LIBCPP_STD_VER >= 23
  template <class _Fp, class... _Args>
  constexpr __optional_storage_base(__optional_construct_from_invoke_tag, _Fp&& __f, _Args&&... __args) {
    _Tp& __r = std::invoke(std::forward<_Fp>(__f), std::forward<_Args>(__args)...);
    __value_ = std::addressof(__r);
  }
#  endif

  _LIBCPP_CONSTEXPR_SINCE_CXX20 void reset() noexcept { __value_ = nullptr; }

  [[nodiscard]] constexpr bool has_value() const noexcept { return __value_ != nullptr; }

  constexpr value_type& __get() const noexcept { return *__value_; }

  template <class _UArg>
  _LIBCPP_CONSTEXPR_SINCE_CXX20 void __construct(_UArg&& __val) {
    _LIBCPP_ASSERT_INTERNAL(!has_value(), "__construct called for engaged __optional_storage");
    static_assert(!__reference_constructs_from_temporary_v<_Tp, _UArg>,
                  "Attempted to construct a reference element in tuple from a "
                  "possible temporary");
    __convert_init_ref_val(std::forward<_UArg>(__val));
  }

  template <class _That>
  _LIBCPP_CONSTEXPR_SINCE_CXX20 void __construct_from(_That&& __opt) {
    if (__opt.has_value())
      __construct(std::forward<_That>(__opt).__get());
  }

  template <class _That>
  _LIBCPP_CONSTEXPR_SINCE_CXX20 void __assign_from(_That&& __opt) {
    if (has_value() == __opt.has_value()) {
      if (has_value())
        *__value_ = std::forward<_That>(__opt).__get();
    } else {
      if (has_value())
        reset();
      else
        __construct(std::forward<_That>(__opt).__get());
    }
  }

  _LIBCPP_CONSTEXPR_SINCE_CXX20 void __swap(__optional_storage_base& __rhs) noexcept {
    std::swap(__value_, __rhs.__value_);
  }

  // [optional.ref.observe]
  constexpr add_pointer_t<_Tp> operator->() const noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(this->has_value(), "optional operator-> called on a disengaged value");
    return std::addressof(this->__get());
  }

  [[nodiscard]] constexpr _Tp& operator*() const noexcept {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(this->has_value(), "optional operator* called on a disengaged value");
    return this->__get();
  }

  [[nodiscard]] constexpr _Tp& value() const {
    if (!this->has_value())
      std::__throw_bad_optional_access();
    return this->__get();
  }
};

template <class _Tp, bool = is_trivially_copy_constructible_v<_Tp> || is_lvalue_reference_v<_Tp>>
struct __optional_copy_base : __optional_storage_base<_Tp> {
  using __optional_storage_base<_Tp>::__optional_storage_base;
};

template <class _Tp>
struct __optional_copy_base<_Tp, false> : __optional_storage_base<_Tp> {
  using __optional_storage_base<_Tp>::__optional_storage_base;

  __optional_copy_base() = default;

  _LIBCPP_CONSTEXPR_SINCE_CXX20 __optional_copy_base(const __optional_copy_base& __opt) {
    this->__construct_from(__opt);
  }

  __optional_copy_base(__optional_copy_base&&)                 = default;
  __optional_copy_base& operator=(const __optional_copy_base&) = default;
  __optional_copy_base& operator=(__optional_copy_base&&)      = default;
};

template <class _Tp, bool = is_trivially_move_constructible_v<_Tp> || is_lvalue_reference_v<_Tp>>
struct __optional_move_base : __optional_copy_base<_Tp> {
  using __optional_copy_base<_Tp>::__optional_copy_base;
};

template <class _Tp>
struct __optional_move_base<_Tp, false> : __optional_copy_base<_Tp> {
  using value_type = _Tp;
  using __optional_copy_base<_Tp>::__optional_copy_base;

  __optional_move_base()                            = default;
  __optional_move_base(const __optional_move_base&) = default;

  _LIBCPP_CONSTEXPR_SINCE_CXX20
  __optional_move_base(__optional_move_base&& __opt) noexcept(is_nothrow_move_constructible_v<value_type>) {
    this->__construct_from(std::move(__opt));
  }

  __optional_move_base& operator=(const __optional_move_base&) = default;
  __optional_move_base& operator=(__optional_move_base&&)      = default;
};

template <class _Tp,
          bool = (is_trivially_destructible_v<_Tp> && is_trivially_copy_constructible_v<_Tp> &&
                  is_trivially_copy_assignable_v<_Tp>) ||
                 is_lvalue_reference_v<_Tp>>
struct __optional_copy_assign_base : __optional_move_base<_Tp> {
  using __optional_move_base<_Tp>::__optional_move_base;
};

template <class _Tp>
struct __optional_copy_assign_base<_Tp, false> : __optional_move_base<_Tp> {
  using __optional_move_base<_Tp>::__optional_move_base;

  __optional_copy_assign_base()                                   = default;
  __optional_copy_assign_base(const __optional_copy_assign_base&) = default;
  __optional_copy_assign_base(__optional_copy_assign_base&&)      = default;

  _LIBCPP_CONSTEXPR_SINCE_CXX20 __optional_copy_assign_base& operator=(const __optional_copy_assign_base& __opt) {
    this->__assign_from(__opt);
    return *this;
  }

  __optional_copy_assign_base& operator=(__optional_copy_assign_base&&) = default;
};

template <class _Tp,
          bool = (is_trivially_destructible_v<_Tp> && is_trivially_move_constructible_v<_Tp> &&
                  is_trivially_move_assignable_v<_Tp>) ||
                 is_lvalue_reference_v<_Tp>>
struct __optional_move_assign_base : __optional_copy_assign_base<_Tp> {
  using __optional_copy_assign_base<_Tp>::__optional_copy_assign_base;
};

template <class _Tp>
struct __optional_move_assign_base<_Tp, false> : __optional_copy_assign_base<_Tp> {
  using value_type = _Tp;
  using __optional_copy_assign_base<_Tp>::__optional_copy_assign_base;

  __optional_move_assign_base()                                              = default;
  __optional_move_assign_base(const __optional_move_assign_base& __opt)      = default;
  __optional_move_assign_base(__optional_move_assign_base&&)                 = default;
  __optional_move_assign_base& operator=(const __optional_move_assign_base&) = default;

  _LIBCPP_CONSTEXPR_SINCE_CXX20 __optional_move_assign_base& operator=(__optional_move_assign_base&& __opt) noexcept(
      is_nothrow_move_assignable_v<value_type> && is_nothrow_move_constructible_v<value_type>) {
    this->__assign_from(std::move(__opt));
    return *this;
  }
};

template <class _Tp>
using __optional_sfinae_ctor_base_t _LIBCPP_NODEBUG =
    __sfinae_ctor_base< is_copy_constructible<_Tp>::value, is_move_constructible<_Tp>::value >;

template <class _Tp>
using __optional_sfinae_assign_base_t _LIBCPP_NODEBUG =
    __sfinae_assign_base< (is_copy_constructible_v<_Tp> && is_copy_assignable_v<_Tp>),
                          (is_move_constructible_v<_Tp> && is_move_assignable_v<_Tp>)>;

template <class _Tp, class = void>
struct __optional_iterator_base : __optional_move_assign_base<_Tp> {
  using __optional_move_assign_base<_Tp>::__optional_move_assign_base;
};

#  if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_OPTIONAL_ITERATOR

template <class _Tp>
struct __optional_iterator_base<_Tp, enable_if_t<is_object_v<_Tp>>> : __optional_move_assign_base<_Tp> {
private:
  using __pointer _LIBCPP_NODEBUG       = add_pointer_t<_Tp>;
  using __const_pointer _LIBCPP_NODEBUG = add_pointer_t<const _Tp>;

public:
  using __optional_move_assign_base<_Tp>::__optional_move_assign_base;

#    ifdef _LIBCPP_ABI_BOUNDED_ITERATORS_IN_OPTIONAL
  using iterator       = __bounded_iter<__pointer>;
  using const_iterator = __bounded_iter<__const_pointer>;
#    else
  using iterator       = __capacity_aware_iterator<__pointer, optional<_Tp>, 1>;
  using const_iterator = __capacity_aware_iterator<__const_pointer, optional<_Tp>, 1>;
#    endif

  // [optional.iterators], iterator support
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr iterator begin() noexcept {
    auto* __ptr = std::addressof(this->__get());

#    ifdef _LIBCPP_ABI_BOUNDED_ITERATORS_IN_OPTIONAL
    return std::__make_bounded_iter(__ptr, __ptr, __ptr + (this->has_value() ? 1 : 0));
#    else
    return std::__make_capacity_aware_iterator<__pointer, optional<_Tp>, 1>(__ptr);
#    endif
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr const_iterator begin() const noexcept {
    auto* __ptr = std::addressof(this->__get());

#    ifdef _LIBCPP_ABI_BOUNDED_ITERATORS_IN_OPTIONAL
    return std::__make_bounded_iter(__ptr, __ptr, __ptr + (this->has_value() ? 1 : 0));
#    else
    return std::__make_capacity_aware_iterator<__const_pointer, optional<_Tp>, 1>(__ptr);
#    endif
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr iterator end() noexcept {
    return begin() + (this->has_value() ? 1 : 0);
  }
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr const_iterator end() const noexcept {
    return begin() + (this->has_value() ? 1 : 0);
  }
};

template <class _Tp>
struct __optional_iterator_base<_Tp&, enable_if_t<is_object_v<_Tp> && !__is_unbounded_array_v<_Tp> >>
    : __optional_move_assign_base<_Tp&> {
private:
  using __pointer _LIBCPP_NODEBUG = add_pointer_t<_Tp>;

public:
  using __optional_move_assign_base<_Tp&>::__optional_move_assign_base;

#    ifdef _LIBCPP_ABI_BOUNDED_ITERATORS_IN_OPTIONAL
  using iterator = __bounded_iter<__pointer>;
#    else
  using iterator = __capacity_aware_iterator<__pointer, optional<_Tp&>, 1>;
#    endif

  // [optional.ref.iterators], iterator support

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const noexcept {
    auto* __ptr = this->has_value() ? std::addressof(this->__get()) : nullptr;

#    ifdef _LIBCPP_ABI_BOUNDED_ITERATORS_IN_OPTIONAL
    return std::__make_bounded_iter(__ptr, __ptr, __ptr + (this->has_value() ? 1 : 0));
#    else
    return std::__make_capacity_aware_iterator<__pointer, optional<_Tp&>, 1>(__ptr);
#    endif
  }

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto end() const noexcept {
    return begin() + (this->has_value() ? 1 : 0);
  }
};

template <class _Tp>
constexpr bool ranges::enable_view<optional<_Tp>> = true;

template <class _Tp>
constexpr range_format format_kind<optional<_Tp>> = range_format::disabled;

template <class _Tp>
constexpr bool ranges::enable_borrowed_range<optional<_Tp&>> = true;

#  endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_EXPERIMENTAL_OPTIONAL_ITERATOR

template <class _Tp, class... _Args>
inline constexpr bool __is_constructible_for_optional_v = is_constructible_v<_Tp, _Args...>;

template <class _Tp, class... _Args>
struct __is_constructible_for_optional : bool_constant<__is_constructible_for_optional_v<_Tp, _Args...>> {};

template <class _Tp, class _Up, class... _Args>
inline constexpr bool __is_constructible_for_optional_initializer_list_v =
    is_constructible_v<_Tp, initializer_list<_Up>&, _Args...>;

#  if _LIBCPP_STD_VER >= 26
template <class _Tp, class... _Args>
inline constexpr bool __is_constructible_for_optional_v<_Tp&, _Args...> = false;

template <class _Tp, class _Arg>
inline constexpr bool __is_constructible_for_optional_v<_Tp&, _Arg> =
    is_constructible_v<_Tp&, _Arg> && !reference_constructs_from_temporary_v<_Tp&, _Arg>;

template <class _Tp, class _Up, class... _Args>
inline constexpr bool __is_constructible_for_optional_initializer_list_v<_Tp&, _Up, _Args...> = false;
#  endif

#  if _LIBCPP_STD_VER >= 26
template <class _Tp>
static constexpr bool __is_valid_optional_type = std::is_object_v<_Tp> || std::is_lvalue_reference_v<_Tp>;
#    define _LIBCPP_OPTIONAL_MONADIC_ASSERT_MESSAGE "Result of f(value()) should be a valid contained type for optional"
#  else
template <class _Tp>
static constexpr bool __is_valid_optional_type = std::is_object_v<_Tp>;
#    define _LIBCPP_OPTIONAL_MONADIC_ASSERT_MESSAGE "Result of f(value()) should be an object type"
#  endif

template <class _Tp>
struct __is_std_optional : false_type {};

template <class _Tp>
struct __is_std_optional<optional<_Tp>> : true_type {};

template <class _Tp>
optional(_Tp) -> optional<_Tp>;

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP_OPTIONAL_COMMON_H
