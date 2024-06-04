//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This header is unguarded on purpose. This header is an implementation detail of move_only_function.h
// and generates multiple versions of std::move_only_function

#include <__assert>
#include <__config>
#include <__cstddef/nullptr_t.h>
#include <__cstddef/size_t.h>
#include <__functional/invoke.h>
#include <__functional/move_only_function_common.h>
#include <__memory/addressof.h>
#include <__memory/construct_at.h>
#include <__type_traits/decay.h>
#include <__type_traits/invoke.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/is_function.h>
#include <__type_traits/is_member_pointer.h>
#include <__type_traits/is_pointer.h>
#include <__type_traits/is_same.h>
#include <__type_traits/is_trivially_destructible.h>
#include <__type_traits/remove_cvref.h>
#include <__type_traits/remove_pointer.h>
#include <__utility/exchange.h>
#include <__utility/forward.h>
#include <__utility/in_place.h>
#include <__utility/move.h>
#include <__utility/small_buffer.h>
#include <__utility/swap.h>
#include <initializer_list>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#ifndef _LIBCPP_IN_MOVE_ONLY_FUNCTION_H
#  error This header should only be included from move_only_function.h
#endif

#ifndef _LIBCPP_MOVE_ONLY_FUNCTION_CV
#  define _LIBCPP_MOVE_ONLY_FUNCTION_CV
#endif

#ifndef _LIBCPP_MOVE_ONLY_FUNCTION_REF
#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF
#  define _LIBCPP_MOVE_ONLY_FUNCTION_INVOKE_QUALS _LIBCPP_MOVE_ONLY_FUNCTION_CV&
#else
#  define _LIBCPP_MOVE_ONLY_FUNCTION_INVOKE_QUALS _LIBCPP_MOVE_ONLY_FUNCTION_CV _LIBCPP_MOVE_ONLY_FUNCTION_REF
#endif

#define _LIBCPP_MOVE_ONLY_FUNCTION_CVREF _LIBCPP_MOVE_ONLY_FUNCTION_CV _LIBCPP_MOVE_ONLY_FUNCTION_REF

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

_LIBCPP_BEGIN_NAMESPACE_STD

template <class...>
class move_only_function;

template <class _ReturnT, class... _ArgTypes, bool __is_noexcept>
class [[_Clang::__trivial_abi__]]
move_only_function<_ReturnT(_ArgTypes...) _LIBCPP_MOVE_ONLY_FUNCTION_CVREF noexcept(__is_noexcept)> {
private:
  static constexpr size_t __buffer_size_      = 3 * sizeof(void*);
  static constexpr size_t __buffer_alignment_ = alignof(void*);
  using _BufferT _LIBCPP_NODEBUG              = __small_buffer<__buffer_size_, __buffer_alignment_>;

  using _VTable _LIBCPP_NODEBUG = _MoveOnlyFunctionVTable<_BufferT, _ReturnT, _ArgTypes...>;

  template <class _Functor>
  static constexpr _VTable __vtable_var_ = {
      .__call_ = [](_BufferT& __buffer, _ArgTypes... __args) noexcept(__is_noexcept) -> _ReturnT {
        return std::invoke_r<_ReturnT>(
            static_cast<_Functor _LIBCPP_MOVE_ONLY_FUNCTION_INVOKE_QUALS>(*__buffer.__get<_Functor>()),
            std::forward<_ArgTypes>(__args)...);
      },
      .__destroy_ = (_BufferT::__fits_in_buffer<_Functor> && is_trivially_destructible_v<_Functor>)
                      ? nullptr
                      : [](_BufferT& __buffer) noexcept -> void {
        std::destroy_at(__buffer.__get<_Functor>());
        __buffer.__dealloc<_Functor>();
      }};

  template <class _VT>
  static constexpr bool __is_callable_from = [] {
    using _DVT = decay_t<_VT>;
    if constexpr (__is_noexcept) {
      return is_nothrow_invocable_r_v<_ReturnT, _DVT _LIBCPP_MOVE_ONLY_FUNCTION_CVREF, _ArgTypes...> &&
             is_nothrow_invocable_r_v<_ReturnT, _DVT _LIBCPP_MOVE_ONLY_FUNCTION_INVOKE_QUALS, _ArgTypes...>;
    } else {
      return is_invocable_r_v<_ReturnT, _DVT _LIBCPP_MOVE_ONLY_FUNCTION_CVREF, _ArgTypes...> &&
             is_invocable_r_v<_ReturnT, _DVT _LIBCPP_MOVE_ONLY_FUNCTION_INVOKE_QUALS, _ArgTypes...>;
    }
  }();

  template <class _Func, class... _Args>
  _LIBCPP_HIDE_FROM_ABI void __construct(_Args&&... __args) {
    static_assert(is_constructible_v<decay_t<_Func>, _Func>);

    using _StoredFunc = decay_t<_Func>;
    __vtable_         = std::addressof(__vtable_var_<_StoredFunc>);
    __buffer_.__construct<_StoredFunc>(std::forward<_Args>(__args)...);
  }

  _LIBCPP_HIDE_FROM_ABI void __reset() noexcept {
    if (__vtable_ && __vtable_->__destroy_)
      __vtable_->__destroy_(__buffer_);
    __vtable_ = nullptr;
  }

public:
  using result_type = _ReturnT;

  // [func.wrap.move.ctor]
  move_only_function() noexcept = default;
  _LIBCPP_HIDE_FROM_ABI move_only_function(nullptr_t) noexcept {}
  _LIBCPP_HIDE_FROM_ABI move_only_function(move_only_function&& __other) noexcept
      : __vtable_(__other.__vtable_), __buffer_(std::move(__other.__buffer_)) {
    __other.__vtable_ = nullptr;
  }

  template <class _Func>
    requires(!is_same_v<remove_cvref_t<_Func>, move_only_function> && !__is_inplace_type<_Func>::value &&
             __is_callable_from<_Func>)
  _LIBCPP_HIDE_FROM_ABI move_only_function(_Func&& __func) noexcept {
    using _StoredFunc = decay_t<_Func>;

    if constexpr ((is_pointer_v<_StoredFunc> && is_function_v<remove_pointer_t<_StoredFunc>>) ||
                  is_member_function_pointer_v<_StoredFunc>) {
      if (__func != nullptr) {
        __vtable_ = std::addressof(__vtable_var_<_StoredFunc>);
        static_assert(_BufferT::__fits_in_buffer<_StoredFunc>);
        __buffer_.__construct<_StoredFunc>(std::forward<_Func>(__func));
      }
    } else if constexpr (__is_move_only_function_v<_StoredFunc>) {
      if (__func) {
        __vtable_ = std::exchange(__func.__vtable_, nullptr);
        __buffer_ = std::move(__func.__buffer_);
      }
    } else {
      __construct<_Func>(std::forward<_Func>(__func));
    }
  }

  template <class _Func, class... _Args>
    requires is_constructible_v<decay_t<_Func>, _Args...> && __is_callable_from<_Func>
  _LIBCPP_HIDE_FROM_ABI explicit move_only_function(in_place_type_t<_Func>, _Args&&... __args) {
    static_assert(is_same_v<decay_t<_Func>, _Func>);
    __construct<_Func>(std::forward<_Args>(__args)...);
  }

  template <class _Func, class _InitListType, class... _Args>
    requires is_constructible_v<decay_t<_Func>, initializer_list<_InitListType>&, _Args...> && __is_callable_from<_Func>
  _LIBCPP_HIDE_FROM_ABI explicit move_only_function(
      in_place_type_t<_Func>, initializer_list<_InitListType> __il, _Args&&... __args) {
    static_assert(is_same_v<decay_t<_Func>, _Func>);
    __construct<_Func>(__il, std::forward<_Args>(__args)...);
  }

  _LIBCPP_HIDE_FROM_ABI move_only_function& operator=(move_only_function&& __other) noexcept {
    move_only_function(std::move(__other)).swap(*this);
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI move_only_function& operator=(nullptr_t) noexcept {
    __reset();
    return *this;
  }

  template <class _Func>
    requires(!is_same_v<remove_cvref_t<_Func>, move_only_function> && !__is_inplace_type<_Func>::value &&
             __is_callable_from<_Func>)
  _LIBCPP_HIDE_FROM_ABI move_only_function& operator=(_Func&& __func) {
    move_only_function(std::forward<_Func>(__func)).swap(*this);
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI ~move_only_function() { __reset(); }

  // [func.wrap.move.inv]
  _LIBCPP_HIDE_FROM_ABI explicit operator bool() const noexcept { return __vtable_; }

  _LIBCPP_HIDE_FROM_ABI _ReturnT operator()(_ArgTypes... __args) _LIBCPP_MOVE_ONLY_FUNCTION_CVREF
      noexcept(__is_noexcept) {
    _LIBCPP_ASSERT_VALID_ELEMENT_ACCESS(static_cast<bool>(*this), "Tried to call a disengaged move_only_function");
    const auto __call = static_cast<_ReturnT (*)(_BufferT&, _ArgTypes...)>(__vtable_->__call_);
    return __call(__buffer_, std::forward<_ArgTypes>(__args)...);
  }

  // [func.wrap.move.util]
  _LIBCPP_HIDE_FROM_ABI void swap(move_only_function& __other) noexcept {
    std::swap(__vtable_, __other.__vtable_);
    std::swap(__buffer_, __other.__buffer_);
  }

  _LIBCPP_HIDE_FROM_ABI friend void swap(move_only_function& __lhs, move_only_function& __rhs) noexcept {
    __lhs.swap(__rhs);
  }

  _LIBCPP_HIDE_FROM_ABI friend bool operator==(const move_only_function& __func, nullptr_t) noexcept { return !__func; }

private:
  const _VTable* __vtable_ = nullptr;
  mutable _BufferT __buffer_;

  template <class...>
  friend class move_only_function;
};

#undef _LIBCPP_MOVE_ONLY_FUNCTION_CV
#undef _LIBCPP_MOVE_ONLY_FUNCTION_REF
#undef _LIBCPP_MOVE_ONLY_FUNCTION_INVOKE_QUALS
#undef _LIBCPP_MOVE_ONLY_FUNCTION_CVREF

_LIBCPP_END_NAMESPACE_STD

_LIBCPP_POP_MACROS
