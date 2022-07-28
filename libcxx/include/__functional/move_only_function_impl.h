//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This header is only partially guarded on purpose. This header is an implementation detail of move_only_function.h
// and generates multiple versions of std::move_only_function

#include <__bit/bit_cast.h>
#include <__config>
#include <__functional/invoke.h>
#include <__memory/construct_at.h>
#include <__memory/unique_ptr.h>
#include <__type_traits/is_trivially_relocatable.h>
#include <__utility/forward.h>
#include <__utility/in_place.h>
#include <__utility/move.h>
#include <climits>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <new>

#include <cassert>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#ifndef _LIBCPP_IN_MOVE_ONLY_FUNCTION_H
#  error This header should only be included from move_only_function.h
#endif

#ifndef _LIBCPP___FUNCTIONAL_MOVE_ONLY_FUNCTION_IMPL_H
#  define _LIBCPP___FUNCTIONAL_MOVE_ONLY_FUNCTION_IMPL_H

_LIBCPP_BEGIN_NAMESPACE_STD

// The storage for move_only_functions may be one of 3 things:
// - A pointer to a heap object
// - A trivially relocatable object
// - A trivially relocatable object that also has a trivial destructor
//
// If the object is too large or not trivially relocatable it is stored on the heap.
// (on 64-bit platforms) an object fits into the small buffer if
// - it is trivially relocatable and has a size of less than 32 bytes and an alignment of no more than 8 bytes, or
// - it is trivially destructible and has a size of less than 40 bytes and an alignment of no more than 8 bytes.
//
// "size of" in this case doesn't mean "sizeof". Objects which have padding bytes at the end may also be put into
// the small buffer if the padding byte can be detected by [[no_unique_address]]. For example
//
//   class [[clang::trivial_abi]] A {
//     int* begin;
//     int* end;
//     int* cap;
//     int flags;
//   public:
//     A(A&&);
//     ~A();
//     void operator()() {}
//   };
//
// is put into the small buffer, even though sizeof(A) == 32.

union __move_only_function_storage {
  using _DestroyFn = void(void*);
  using _CallFn    = void;

  static constexpr auto __target_size = 6 * sizeof(void*);

  enum class _Status : uint8_t {
    _NotEngaged,
    _TriviallyRelocatable,
    _TriviallyDestructible,
    _Heap,
  };

  struct _HeapObject {
    _CallFn* __call_;
    _DestroyFn* __destroy_;
    std::byte* __heap_;
    std::byte __padding_[__target_size - 3 * sizeof(void*) - 1];
    _Status __status_;
  } __large_;
  static_assert(sizeof(_HeapObject) == __target_size);
  static_assert(offsetof(_HeapObject, __status_) == __target_size - 1);

  struct _TriviallyRelocatableObject {
    _CallFn* __call_;
    _DestroyFn* __destroy_;
    alignas(void*) std::byte __data_[__target_size - 2 * sizeof(void*) - 1];
    _Status __status_;
  } __trivially_relocatable_;
  static_assert(sizeof(_TriviallyRelocatableObject) == __target_size);
  static_assert(offsetof(_TriviallyRelocatableObject, __status_) == __target_size - 1);

  struct _TriviallyDestructibleObject {
    _CallFn* __call_;
    alignas(void*) std::byte __data_[__target_size - 1 * sizeof(void*) - 1];
    _Status __status_;
  } __trivially_destructible_;
  static_assert(sizeof(_TriviallyDestructibleObject) == __target_size);
  static_assert(offsetof(_TriviallyDestructibleObject, __status_) == __target_size - 1);

private:
  // try to squeeze in another char to see if we can use the last byte for our own purposes
  template <class _Func>
  struct _Compact {
    _LIBCPP_NO_UNIQUE_ADDRESS _Func __func;
    char __c;
  };

  template <class _Func, size_t _BufferSize, size_t BufferAlignment>
  static constexpr bool __fits_in_buffer =
      sizeof(_Compact<_Func>) <= _BufferSize + 1 && alignof(_Func) <= BufferAlignment;

  template <class _Func>
  static constexpr bool __fits_in_trivially_relocatable_buffer_impl =
      __libcpp_is_trivially_relocatable_v<_Func> &&
      __fits_in_buffer<_Func,
                       sizeof(_TriviallyRelocatableObject::__data_),
                       alignof(_TriviallyRelocatableObject::__data_)>;

  template <class _Func>
  static constexpr bool __fits_in_trivially_destructible_buffer_impl =
      __libcpp_is_trivially_relocatable_v<_Func> && is_trivially_destructible_v<_Func> &&
      __fits_in_buffer<_Func,
                       sizeof(_TriviallyDestructibleObject::__data_),
                       alignof(_TriviallyDestructibleObject::__data_)>;

public:
  template <class _Func>
  static constexpr bool __fits_in_trivially_relocatable_buffer =
      __fits_in_trivially_relocatable_buffer_impl<remove_cvref_t<_Func>>;

  template <class _Func>
  static constexpr bool __fits_in_trivially_destructible_buffer =
      __fits_in_trivially_destructible_buffer_impl<remove_cvref_t<_Func>>;
};

static_assert(alignof(__move_only_function_storage) == alignof(void*));
static_assert(sizeof(__move_only_function_storage) == __move_only_function_storage::__target_size);

template <class...>
class move_only_function;

template <class...>
struct __is_move_only_function : false_type {};

template <class _ReturnT, class... _ArgTypes>
struct __is_move_only_function<move_only_function<_ReturnT(_ArgTypes...)>> : true_type {};

_LIBCPP_END_NAMESPACE_STD

// This header is only partially guarded on purpose. This header is an implementation detail of move_only_function.h
// and generates multiple versions of std::move_only_function
#endif // _LIBCPP___FUNCTIONAL_MOVE_ONLY_FUNCTION_IMPL_H

#ifndef _LIBCPP_MOVE_ONLY_FUNCTION_CV
#  define _LIBCPP_MOVE_ONLY_FUNCTION_CV
#endif

#ifndef _LIBCPP_MOVE_ONLY_FUNCTION_REF
#  define _LIBCPP_MOVE_ONLY_FUNCTION_REF
#  define _LIBCPP_MOVE_ONLY_FUNCTION_INV_QUALS _LIBCPP_MOVE_ONLY_FUNCTION_CV&
#else
#  define _LIBCPP_MOVE_ONLY_FUNCTION_INV_QUALS _LIBCPP_MOVE_ONLY_FUNCTION_CV _LIBCPP_MOVE_ONLY_FUNCTION_REF
#endif

#ifndef _LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT
#  define _LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT false
#endif

#define _LIBCPP_MOVE_ONLY_FUNCTION_CV_REF _LIBCPP_MOVE_ONLY_FUNCTION_CV _LIBCPP_MOVE_ONLY_FUNCTION_REF

_LIBCPP_BEGIN_NAMESPACE_STD

#ifdef _LIBCPP_ABI_MOVE_ONLY_FUNCTION_TRIVIAL_ABI
#  define _LIBCPP_MOVE_ONLY_FUNCTION_TRIVIAL_ABI [[_Clang::__trivial_abi__]]
#else
#  define _LIBCPP_MOVE_ONLY_FUNCTION_TRIVIAL_ABI
#endif

template <class _ReturnT, class... _ArgTypes>
class _LIBCPP_MOVE_ONLY_FUNCTION_TRIVIAL_ABI move_only_function<_ReturnT(
    _ArgTypes...) _LIBCPP_MOVE_ONLY_FUNCTION_CV_REF noexcept(_LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT)> {
  using _Store = __move_only_function_storage;

  template <class _Functor>
  struct __function_wrappers_impl {
    __function_wrappers_impl() = delete;
    __function_wrappers_impl(const __function_wrappers_impl&) = delete;
    __function_wrappers_impl(__function_wrappers_impl&&)      = delete;

    _LIBCPP_HIDE_FROM_ABI static void __destroy(void* __store) noexcept {
      std::destroy_at(static_cast<_Functor*>(__store));
    }

    _LIBCPP_HIDE_FROM_ABI static _ReturnT
    __call(void* __functor, _ArgTypes&&... __args) noexcept(_LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT) {
      return std::invoke(static_cast<_Functor _LIBCPP_MOVE_ONLY_FUNCTION_INV_QUALS>(*static_cast<_Functor*>(__functor)),
                         std::forward<_ArgTypes>(__args)...);
    }
  };

  template <class _Functor>
  using __function_wrappers = __function_wrappers_impl<remove_cvref_t<_Functor>>;

  template <class _VT>
  consteval static bool __is_callable_from_impl() {
    if (_LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT) {
      return is_nothrow_invocable_r_v< _ReturnT, _VT _LIBCPP_MOVE_ONLY_FUNCTION_CV_REF, _ArgTypes...> &&
             is_nothrow_invocable_r_v<_ReturnT, _VT _LIBCPP_MOVE_ONLY_FUNCTION_INV_QUALS, _ArgTypes...>;
    } else {
      return is_invocable_r_v<_ReturnT, _VT _LIBCPP_MOVE_ONLY_FUNCTION_CV_REF, _ArgTypes...> &&
             is_invocable_r_v<_ReturnT, _VT _LIBCPP_MOVE_ONLY_FUNCTION_INV_QUALS, _ArgTypes...>;
    }
    return false;
  }

  template <class _VT>
  static constexpr bool __is_callable_from = __is_callable_from_impl<decay_t<_VT>>();

  template <class _Func, class... _Args>
  _LIBCPP_HIDE_FROM_ABI void __construct(_Args&&... __args) {
    static_assert(is_constructible_v<decay_t<_Func>, _Func>);

    using _FuncWraps = __function_wrappers<_Func>;
    using _UnRefFunc = remove_reference_t<_Func>;

    if constexpr (_Store::__fits_in_trivially_destructible_buffer<_Func>) {
      auto& __data   = __storage_.__trivially_destructible_;
      __data.__call_ = std::bit_cast<void*>(&_FuncWraps::__call);
      std::construct_at(reinterpret_cast<_UnRefFunc*>(__data.__data_), std::forward<_Args>(__args)...);
      __data.__status_ = _Store::_Status::_TriviallyDestructible;
    } else if constexpr (_Store::__fits_in_trivially_relocatable_buffer<_Func>) {
      auto& __data      = __storage_.__trivially_relocatable_;
      __data.__call_    = std::bit_cast<void*>(&_FuncWraps::__call);
      __data.__destroy_ = &_FuncWraps::__destroy;
      std::construct_at(reinterpret_cast<_UnRefFunc*>(__data.__data_), std::forward<_Args>(__args)...);
      __data.__status_ = _Store::_Status::_TriviallyRelocatable;
    } else {
      auto& __data      = __storage_.__large_;
      __data.__call_    = std::bit_cast<void*>(&_FuncWraps::__call);
      __data.__destroy_ = &_FuncWraps::__destroy;
      __data.__heap_    = static_cast<std::byte*>(::operator new(sizeof(_Func), std::align_val_t(alignof(_Func))));
      std::construct_at(reinterpret_cast<_UnRefFunc*>(__data.__heap_), std::forward<_Args>(__args)...);
      __data.__status_ = _Store::_Status::_Heap;
    }
  }

  _LIBCPP_HIDE_FROM_ABI void __reset() {
    using enum _Store::_Status;
    switch (__storage_.__large_.__status_) {
    case _NotEngaged:
    case _TriviallyDestructible:
      break;

    case _TriviallyRelocatable:
      __storage_.__trivially_relocatable_.__destroy_(&__storage_.__trivially_relocatable_.__data_);
      break;
    case _Heap:
      __storage_.__large_.__destroy_(__storage_.__large_.__heap_);
      ::operator delete(__storage_.__large_.__heap_);
      break;
    }
    __storage_.__large_.__status_ = _NotEngaged;
  }

public:
  using result_type = _ReturnT;

  // [func.wrap.move.ctor]
  move_only_function() noexcept = default;
  _LIBCPP_HIDE_FROM_ABI move_only_function(nullptr_t) noexcept : __storage_({}) {}
  _LIBCPP_HIDE_FROM_ABI move_only_function(move_only_function&& __other) noexcept : __storage_(__other.__storage_) {
    __other.__storage_ = {};
  }

  template <class _Func>
    requires(!is_same_v<remove_cvref_t<_Func>, move_only_function> && !__is_inplace_type<_Func>::value &&
             __is_callable_from<_Func>)
  _LIBCPP_HIDE_FROM_ABI move_only_function(_Func&& __func) noexcept {
    using _FuncWraps = __function_wrappers<_Func>;
    using _UnRefFunc = remove_reference_t<_Func>;

    if constexpr ((is_pointer_v<_UnRefFunc> && is_function_v<remove_pointer_t<_UnRefFunc>>) ||
                  is_member_function_pointer_v<_UnRefFunc>) {
      if (__func == nullptr) {
        __storage_ = {};
        return;
      } else {
        auto& __data   = __storage_.__trivially_destructible_;
        __data.__call_ = std::bit_cast<void*>(&_FuncWraps::__call);
        std::construct_at(reinterpret_cast<_UnRefFunc*>(__data.__data_), std::forward<_Func>(__func));
        __data.__status_ = _Store::_Status::_TriviallyDestructible;
      }
    } else if constexpr (__is_move_only_function<remove_cvref_t<_Func>>::value) {
      if (!__func) {
        __storage_ = {};
      } else {
        auto& __data      = __storage_.__large_;
        __data.__call_    = std::bit_cast<void*>(&_FuncWraps::__call);
        __data.__destroy_ = &_FuncWraps::__destroy;
        __data.__heap_    = static_cast<std::byte*>(::operator new(sizeof(_Func), std::align_val_t(alignof(_Func))));
        std::construct_at(reinterpret_cast<_UnRefFunc*>(__data.__heap_), std::forward<_Func>(__func));
        __data.__status_ = _Store::_Status::_Heap;
      }
    } else {
      __construct<_Func>(std::forward<_Func>(__func));
    }
  }

  _LIBCPP_HIDE_FROM_ABI _Store::_Status __get_status() noexcept { return __storage_.__large_.__status_; }

  template <class _Func, class... _Args>
    requires is_constructible_v<decay_t<_Func>, _Args...> && __is_callable_from<_Func>
  _LIBCPP_HIDE_FROM_ABI explicit move_only_function(in_place_type_t<_Func>, _Args&&... __args) {
    static_assert(is_same_v<decay_t<_Func>, _Func>);
    __construct<_Func>(std::forward<_Args>(__args)...);
  }

  template <class _Func, class _InitListType, class... _Args>
    requires is_constructible_v<decay_t<_Func>, initializer_list<_InitListType>, _Args...> && __is_callable_from<_Func>
  _LIBCPP_HIDE_FROM_ABI explicit move_only_function(
      in_place_type_t<_Func>, initializer_list<_InitListType> __il, _Args&&... __args) {
    __construct<_Func>(__il, std::forward<_Args>(__args)...);
  }

  // TODO: Do we want to make this `noexcept` as an extensions?
  _LIBCPP_HIDE_FROM_ABI move_only_function& operator=(move_only_function&& __other) noexcept {
    move_only_function(std::move(__other)).swap(*this);
    return *this;
  }

  _LIBCPP_HIDE_FROM_ABI move_only_function& operator=(nullptr_t) noexcept {
    __reset();
    return *this;
  }

  // TODO: Do we want to make this `noexcept` as an extensions?
  template <class _Func>
  _LIBCPP_HIDE_FROM_ABI move_only_function& operator=(_Func&& __func) noexcept {
    move_only_function(std::forward<_Func>(__func)).swap(*this);
  }

  _LIBCPP_HIDE_FROM_ABI ~move_only_function() { __reset(); }

  // [func.wrap.move.inv]
  _LIBCPP_HIDE_FROM_ABI explicit operator bool() const noexcept {
    return __storage_.__large_.__status_ != _Store::_Status::_NotEngaged;
  }

  _LIBCPP_HIDE_FROM_ABI _ReturnT operator()(_ArgTypes... __args) _LIBCPP_MOVE_ONLY_FUNCTION_CV_REF
      noexcept(_LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT) {
    auto __call = [&](auto&& __callable, auto&& __data) {
      return std::bit_cast<void (*)(_LIBCPP_MOVE_ONLY_FUNCTION_CV void*, _ArgTypes...)>(__callable)(
          __data, std::forward<_ArgTypes>(__args)...);
    };

    using enum _Store::_Status;
    switch (__storage_.__large_.__status_) {
    case _NotEngaged:
      _LIBCPP_ASSERT(false, "Tried to call move_only_function which doesn't hold a callable object");
    case _TriviallyDestructible:
      return __call(__storage_.__trivially_destructible_.__call_, &__storage_.__trivially_destructible_.__data_);
    case _TriviallyRelocatable:
      return __call(__storage_.__trivially_relocatable_.__call_, &__storage_.__trivially_relocatable_.__data_);
    case _Heap:
      return __call(__storage_.__large_.__call_, &__storage_.__large_.__heap_);
    }
  }

  // [func.wrap.move.util]
  _LIBCPP_HIDE_FROM_ABI void swap(move_only_function& __other) noexcept { std::swap(__storage_, __other.__storage_); }

  _LIBCPP_HIDE_FROM_ABI friend void swap(move_only_function& __lhs, move_only_function& __rhs) noexcept {
    __lhs.swap(__rhs);
  }

  _LIBCPP_HIDE_FROM_ABI friend bool operator==(const move_only_function& __func, nullptr_t) noexcept { return !__func; }

private:
  _Store __storage_ = {};
};

template <class _ReturnT, class... _Args>
struct __libcpp_is_trivially_relocatable<move_only_function<_ReturnT(
    _Args...) _LIBCPP_MOVE_ONLY_FUNCTION_CV_REF noexcept(_LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT)>> : true_type {};

#undef _LIBCPP_MOVE_ONLY_FUNCTION_CV
#undef _LIBCPP_MOVE_ONLY_FUNCTION_REF
#undef _LIBCPP_MOVE_ONLY_FUNCTION_NOEXCEPT
#undef _LIBCPP_MOVE_ONLY_FUNCTION_INV_QUALS
#undef _LIBCPP_MOVE_ONLY_FUNCTION_CV_REF

_LIBCPP_END_NAMESPACE_STD
