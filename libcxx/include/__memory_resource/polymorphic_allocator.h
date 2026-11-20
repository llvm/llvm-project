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
#include <__memory/uses_allocator_construction.h>
#include <__memory_resource/memory_resource.h>
#include <__new/exceptions.h>
#include <__utility/exception_guard.h>
#include <__utility/forward.h>
#include <limits>

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

  polymorphic_allocator() noexcept : __res_(std::pmr::get_default_resource()) {}

  polymorphic_allocator(memory_resource* _LIBCPP_DIAGNOSE_NULLPTR __r) noexcept : __res_(__r) {
    _LIBCPP_ASSERT_NON_NULL(__r, "Attempted to pass a nullptr resource to polymorphic_alloator");
  }

  polymorphic_allocator(const polymorphic_allocator&) = default;

  template <class _Tp>
  polymorphic_allocator(const polymorphic_allocator<_Tp>& __other) noexcept : __res_(__other.resource()) {}

  polymorphic_allocator& operator=(const polymorphic_allocator&) = delete;

  // [mem.poly.allocator.mem]

  [[nodiscard]] _ValueType* allocate(size_t __n) {
    if (__n > __max_size()) {
      std::__throw_bad_array_new_length();
    }
    return static_cast<_ValueType*>(__res_->allocate(__n * sizeof(_ValueType), alignof(_ValueType)));
  }

  void deallocate(_ValueType* __p, size_t __n) {
    _LIBCPP_ASSERT_VALID_DEALLOCATION(
        __n <= __max_size(),
        "deallocate() called for a size which exceeds max_size(), leading to a memory leak "
        "(the argument will overflow and result in too few objects being deleted)");
    __res_->deallocate(__p, __n * sizeof(_ValueType), alignof(_ValueType));
  }

#  if _LIBCPP_STD_VER >= 20

  [[nodiscard]] [[using __gnu__: __alloc_size__(2), __alloc_align__(3)]] void*
  allocate_bytes(size_t __nbytes, size_t __alignment = alignof(max_align_t)) {
    return __res_->allocate(__nbytes, __alignment);
  }

  void deallocate_bytes(void* __ptr, size_t __nbytes, size_t __alignment = alignof(max_align_t)) {
    __res_->deallocate(__ptr, __nbytes, __alignment);
  }

  template <class _Type>
  [[nodiscard]] _Type* allocate_object(size_t __n = 1) {
    if (numeric_limits<size_t>::max() / sizeof(_Type) < __n)
      std::__throw_bad_array_new_length();
    return static_cast<_Type*>(allocate_bytes(__n * sizeof(_Type), alignof(_Type)));
  }

  template <class _Type>
  void deallocate_object(_Type* __ptr, size_t __n = 1) {
    deallocate_bytes(__ptr, __n * sizeof(_Type), alignof(_Type));
  }

  template <class _Type, class... _CtorArgs>
  [[nodiscard]] _Type* new_object(_CtorArgs&&... __ctor_args) {
    _Type* __ptr = allocate_object<_Type>();
    auto __guard = std::__make_exception_guard([&] { deallocate_object(__ptr); });
    construct(__ptr, std::forward<_CtorArgs>(__ctor_args)...);
    __guard.__complete();
    return __ptr;
  }

  template <class _Type>
  void delete_object(_Type* __ptr) {
    destroy(__ptr);
    deallocate_object(__ptr);
  }

#  endif // _LIBCPP_STD_VER >= 20

  template <class _Tp, class... _Args>
  void construct(_Tp* __p, _Args&&... __args) {
    std::__uninitialized_construct_using_allocator(__p, *this, std::forward<_Args>(__args)...);
  }

  template <class _Tp>
  void destroy(_Tp* __p) {
    __p->~_Tp();
  }

  [[nodiscard]] polymorphic_allocator select_on_container_copy_construction() const noexcept {
    return polymorphic_allocator();
  }

  [[nodiscard, __gnu__::__returns_nonnull__]] memory_resource* resource() const noexcept { return __res_; }

  friend bool operator==(const polymorphic_allocator& __lhs, const polymorphic_allocator& __rhs) noexcept {
    return *__lhs.resource() == *__rhs.resource();
  }

#  if _LIBCPP_STD_VER <= 17
  // This overload is not specified, it was added due to LWG3683.
  friend bool operator!=(const polymorphic_allocator& __lhs, const polymorphic_allocator& __rhs) noexcept {
    return *__lhs.resource() != *__rhs.resource();
  }
#  endif

private:
  size_t __max_size() const noexcept { return numeric_limits<size_t>::max() / sizeof(value_type); }

  memory_resource* __res_;
};

// [mem.poly.allocator.eq]

template <class _Tp, class _Up>
inline bool operator==(const polymorphic_allocator<_Tp>& __lhs, const polymorphic_allocator<_Up>& __rhs) noexcept {
  return *__lhs.resource() == *__rhs.resource();
}

#  if _LIBCPP_STD_VER <= 17

template <class _Tp, class _Up>
inline bool operator!=(const polymorphic_allocator<_Tp>& __lhs, const polymorphic_allocator<_Up>& __rhs) noexcept {
  return !(__lhs == __rhs);
}

#  endif

} // namespace pmr

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 17

_LIBCPP_POP_MACROS

#endif // _LIBCPP___MEMORY_RESOURCE_POLYMORPHIC_ALLOCATOR_H
