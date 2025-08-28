//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_RESOURCE_POLYMORPHIC_ALLOCATOR_H
#define _LIBCPP___MEMORY_RESOURCE_POLYMORPHIC_ALLOCATOR_H

#include <__assert>
#include <__config>
#include <__cstddef/byte.h>
#include <__cstddef/max_align_t.h>
#include <__fwd/pair.h>
#include <__memory/allocator_arg_t.h>
#include <__memory/uses_allocator.h>
#include <__memory/uses_allocator_construction.h>
#include <__memory_resource/memory_resource.h>
#include <__new/exceptions.h>
#include <__new/placement_new_delete.h>
#include <__type_traits/is_constructible.h>
#include <__type_traits/remove_cv.h>
#include <__utility/as_const.h>
#include <__utility/exception_guard.h>
#include <__utility/piecewise_construct.h>
#include <limits>
#include <tuple>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_PUSH_MACROS
#include <__undef_macros>

#if _LIBCPP_STD_VER >= 17

_LIBCPP_BEGIN_NAMESPACE_STD

namespace pmr {

// [mem.poly.allocator.class]

template <class _ValueType
#  if _LIBCPP_STD_VER >= 20
          = byte
#  endif
          >
class _LIBCPP_AVAILABILITY_PMR polymorphic_allocator {

public:
  using value_type = _ValueType;

  // [mem.poly.allocator.ctor]

  _LIBCPP_HIDE_FROM_ABI polymorphic_allocator() noexcept : __res_(std::pmr::get_default_resource()) {}

  _LIBCPP_HIDE_FROM_ABI polymorphic_allocator(memory_resource* _LIBCPP_DIAGNOSE_NULLPTR __r) noexcept : __res_(__r) {
    _LIBCPP_ASSERT_NON_NULL(__r, "Attempted to pass a nullptr resource to polymorphic_alloator");
  }

  _LIBCPP_HIDE_FROM_ABI polymorphic_allocator(const polymorphic_allocator&) = default;

  template <class _Tp>
  _LIBCPP_HIDE_FROM_ABI polymorphic_allocator(const polymorphic_allocator<_Tp>& __other) noexcept
      : __res_(__other.resource()) {}

  polymorphic_allocator& operator=(const polymorphic_allocator&) = delete;

  // [mem.poly.allocator.mem]

  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI _ValueType* allocate(size_t __n) {
    if (__n > __max_size()) {
      std::__throw_bad_array_new_length();
    }
    return static_cast<_ValueType*>(__res_->allocate(__n * sizeof(_ValueType), alignof(_ValueType)));
  }

  _LIBCPP_HIDE_FROM_ABI void deallocate(_ValueType* __p, size_t __n) {
    _LIBCPP_ASSERT_VALID_DEALLOCATION(
        __n <= __max_size(),
        "deallocate() called for a size which exceeds max_size(), leading to a memory leak "
        "(the argument will overflow and result in too few objects being deleted)");
    __res_->deallocate(__p, __n * sizeof(_ValueType), alignof(_ValueType));
  }

#  if _LIBCPP_STD_VER >= 20

  [[nodiscard]] [[using __gnu__: __alloc_size__(2), __alloc_align__(3)]] _LIBCPP_HIDE_FROM_ABI void*
  allocate_bytes(size_t __nbytes, size_t __alignment = alignof(max_align_t)) {
    return __res_->allocate(__nbytes, __alignment);
  }

  _LIBCPP_HIDE_FROM_ABI void deallocate_bytes(void* __ptr, size_t __nbytes, size_t __alignment = alignof(max_align_t)) {
    __res_->deallocate(__ptr, __nbytes, __alignment);
  }

  template <class _Type>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI _Type* allocate_object(size_t __n = 1) {
    if (numeric_limits<size_t>::max() / sizeof(_Type) < __n)
      std::__throw_bad_array_new_length();
    return static_cast<_Type*>(allocate_bytes(__n * sizeof(_Type), alignof(_Type)));
  }

  template <class _Type>
  _LIBCPP_HIDE_FROM_ABI void deallocate_object(_Type* __ptr, size_t __n = 1) {
    deallocate_bytes(__ptr, __n * sizeof(_Type), alignof(_Type));
  }

  template <class _Type, class... _CtorArgs>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI _Type* new_object(_CtorArgs&&... __ctor_args) {
    _Type* __ptr = allocate_object<_Type>();
    auto __guard = std::__make_exception_guard([&] { deallocate_object(__ptr); });
    construct(__ptr, std::forward<_CtorArgs>(__ctor_args)...);
    __guard.__complete();
    return __ptr;
  }

  template <class _Type>
  _LIBCPP_HIDE_FROM_ABI void delete_object(_Type* __ptr) {
    destroy(__ptr);
    deallocate_object(__ptr);
  }

#  endif // _LIBCPP_STD_VER >= 20

  template <class _Tp, class... _Ts>
  _LIBCPP_HIDE_FROM_ABI void construct(_Tp* __p, _Ts&&... __args) {
    if constexpr (!uses_allocator_v<remove_cv_t<_Tp>, polymorphic_allocator>) {
      static_assert(is_constructible_v<_Tp, _Ts...>,
                    "If uses_allocator_v<T, polymorphic_allocator> is false, T has to be constructible from arguments");
      ::new ((void*)__p) _Tp(std::forward<_Ts>(__args)...);
    } else if constexpr (is_constructible_v<_Tp, allocator_arg_t, const polymorphic_allocator&, _Ts...>) {
      ::new ((void*)__p) _Tp(allocator_arg, std::as_const(*this), std::forward<_Ts>(__args)...);
    } else if constexpr (is_constructible_v<_Tp, _Ts..., const polymorphic_allocator&>) {
      ::new ((void*)__p) _Tp(std::forward<_Ts>(__args)..., std::as_const(*this));
    } else {
      static_assert(
          false, "If uses_allocator_v<T, polymorphic_allocator> is true, T has to be allocator-constructible");
    }
  }

  template <class _T1, class _T2, class... _Args1, class... _Args2>
  _LIBCPP_HIDE_FROM_ABI void
  construct(pair<_T1, _T2>* __p, piecewise_construct_t, tuple<_Args1...> __x, tuple<_Args2...> __y) {
    ::new ((void*)__p) pair<_T1, _T2>(
        piecewise_construct,
        std::__transform_tuple_using_allocator<_T1>(*this, std::move(__x)),
        std::__transform_tuple_using_allocator<_T2>(*this, std::move(__y)));
  }

  template <class _T1, class _T2>
  _LIBCPP_HIDE_FROM_ABI void construct(pair<_T1, _T2>* __p) {
    construct(__p, piecewise_construct, tuple<>(), tuple<>());
  }

  template <class _T1, class _T2, class _Up, class _Vp>
  _LIBCPP_HIDE_FROM_ABI void construct(pair<_T1, _T2>* __p, _Up&& __u, _Vp&& __v) {
    construct(__p,
              piecewise_construct,
              std::forward_as_tuple(std::forward<_Up>(__u)),
              std::forward_as_tuple(std::forward<_Vp>(__v)));
  }

  template <class _T1, class _T2, class _U1, class _U2>
  _LIBCPP_HIDE_FROM_ABI void construct(pair<_T1, _T2>* __p, const pair<_U1, _U2>& __pr) {
    construct(__p, piecewise_construct, std::forward_as_tuple(__pr.first), std::forward_as_tuple(__pr.second));
  }

  template <class _T1, class _T2, class _U1, class _U2>
  _LIBCPP_HIDE_FROM_ABI void construct(pair<_T1, _T2>* __p, pair<_U1, _U2>&& __pr) {
    construct(__p,
              piecewise_construct,
              std::forward_as_tuple(std::forward<_U1>(__pr.first)),
              std::forward_as_tuple(std::forward<_U2>(__pr.second)));
  }

  template <class _Tp>
  _LIBCPP_HIDE_FROM_ABI void destroy(_Tp* __p) {
    __p->~_Tp();
  }

  _LIBCPP_HIDE_FROM_ABI polymorphic_allocator select_on_container_copy_construction() const noexcept {
    return polymorphic_allocator();
  }

  [[__gnu__::__returns_nonnull__]] _LIBCPP_HIDE_FROM_ABI memory_resource* resource() const noexcept { return __res_; }

  _LIBCPP_HIDE_FROM_ABI friend bool
  operator==(const polymorphic_allocator& __lhs, const polymorphic_allocator& __rhs) noexcept {
    return *__lhs.resource() == *__rhs.resource();
  }

#  if _LIBCPP_STD_VER <= 17
  // This overload is not specified, it was added due to LWG3683.
  _LIBCPP_HIDE_FROM_ABI friend bool
  operator!=(const polymorphic_allocator& __lhs, const polymorphic_allocator& __rhs) noexcept {
    return *__lhs.resource() != *__rhs.resource();
  }
#  endif

private:
  _LIBCPP_HIDE_FROM_ABI size_t __max_size() const noexcept {
    return numeric_limits<size_t>::max() / sizeof(value_type);
  }

  memory_resource* __res_;
};

// [mem.poly.allocator.eq]

template <class _Tp, class _Up>
inline _LIBCPP_HIDE_FROM_ABI bool
operator==(const polymorphic_allocator<_Tp>& __lhs, const polymorphic_allocator<_Up>& __rhs) noexcept {
  return *__lhs.resource() == *__rhs.resource();
}

#  if _LIBCPP_STD_VER <= 17

template <class _Tp, class _Up>
inline _LIBCPP_HIDE_FROM_ABI bool
operator!=(const polymorphic_allocator<_Tp>& __lhs, const polymorphic_allocator<_Up>& __rhs) noexcept {
  return !(__lhs == __rhs);
}

#  endif

} // namespace pmr

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_RESOURCE_POLYMORPHIC_ALLOCATOR_H
