// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___MEMORY_ATOMIC_SHARED_PTR_H
#define _LIBCPP___MEMORY_ATOMIC_SHARED_PTR_H

#include <__memory/addressof.h>
#include <__memory/shared_ptr.h>
#include <cstddef>
#if !defined(_LIBCPP_HAS_NO_ATOMIC_HEADER)
#  include <__atomic/memory_order.h>
#endif

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if !defined(_LIBCPP_HAS_NO_THREADS)

class _LIBCPP_EXPORTED_FROM_ABI __sp_mut {
  void* __lx_;

public:
  void lock() _NOEXCEPT;
  void unlock() _NOEXCEPT;

private:
  _LIBCPP_CONSTEXPR __sp_mut(void*) _NOEXCEPT;
  __sp_mut(const __sp_mut&);
  __sp_mut& operator=(const __sp_mut&);

  friend _LIBCPP_EXPORTED_FROM_ABI __sp_mut& __get_sp_mut(const void*);
};

_LIBCPP_EXPORTED_FROM_ABI __sp_mut& __get_sp_mut(const void*);

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _Tp __sp_atomic_load(const _Tp* __p) {
  __sp_mut& __m = std::__get_sp_mut(__p);
  __m.lock();
  _Tp __q = *__p;
  __m.unlock();
  return __q;
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI void __sp_atomic_store(_Tp* __p, _Tp& __r) {
  __sp_mut& __m = std::__get_sp_mut(__p);
  __m.lock();
  __p->swap(__r);
  __m.unlock();
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI bool __sp_atomic_compare_exchange_strong(_Tp* __p, _Tp* __v, _Tp& __w) {
  _Tp __temp;
  __sp_mut& __m = std::__get_sp_mut(__p);
  __m.lock();
  if (__p->__owner_equivalent(*__v)) {
    std::swap(__temp, *__p);
    *__p = __w;
    __m.unlock();
    return true;
  }
  std::swap(__temp, *__v);
  *__v = *__p;
  __m.unlock();
  return false;
}

template <class _Tp>
_LIBCPP_HIDE_FROM_ABI _Tp __sp_atomic_exchange(_Tp* __p, _Tp& __r) {
  __sp_mut& __m = std::__get_sp_mut(__p);
  __m.lock();
  __p->swap(__r);
  __m.unlock();
  return __r;
}

#  if _LIBCPP_STD_VER >= 20
template <class _Tp>
struct atomic;

template <class _Tp>
struct __sp_atomic_base {
  using value_type = _Tp;

  static constexpr bool is_always_lock_free = false;
  _LIBCPP_HIDE_FROM_ABI bool is_lock_free() const noexcept { return false; }

  _LIBCPP_HIDE_FROM_ABI constexpr __sp_atomic_base() noexcept = default;
  _LIBCPP_HIDE_FROM_ABI __sp_atomic_base(_Tp&& __d) noexcept : __p(std::move(__d)) {}
  _LIBCPP_HIDE_FROM_ABI __sp_atomic_base(const __sp_atomic_base&) = delete;
  _LIBCPP_HIDE_FROM_ABI void operator=(const __sp_atomic_base&)   = delete;

  _LIBCPP_HIDE_FROM_ABI _Tp load(memory_order = memory_order_seq_cst) noexcept {
    return std::__sp_atomic_load(std::addressof(__p));
  }
  _LIBCPP_HIDE_FROM_ABI operator _Tp() const noexcept { return load(); }
  _LIBCPP_HIDE_FROM_ABI void store(_Tp __d, memory_order = memory_order_seq_cst) noexcept {
    std::__sp_atomic_store(std::addressof(__p), __d);
  }
  _LIBCPP_HIDE_FROM_ABI void operator=(_Tp __d) noexcept { std::__sp_atomic_store(std::addressof(__p), __d); }
  _LIBCPP_HIDE_FROM_ABI void operator=(nullptr_t) noexcept { store(nullptr); }

  _LIBCPP_HIDE_FROM_ABI _Tp exchange(_Tp __d, memory_order = memory_order_seq_cst) noexcept {
    return std::__sp_atomic_exchange(std::addressof(__p), __d);
  }
  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_weak(_Tp& __e, _Tp __d, memory_order, memory_order) noexcept {
    return std::__sp_atomic_compare_exchange_strong(std::addressof(__p), std::addressof(__e), __d);
  }
  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_strong(_Tp& __e, _Tp __d, memory_order, memory_order) noexcept {
    return std::__sp_atomic_compare_exchange_strong(std::addressof(__p), std::addressof(__e), __d);
  }
  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_weak(_Tp& __e, _Tp __d, memory_order = memory_order_seq_cst) noexcept {
    return std::__sp_atomic_compare_exchange_strong(std::addressof(__p), std::addressof(__e), __d);
  }
  _LIBCPP_HIDE_FROM_ABI bool compare_exchange_strong(_Tp& __e, _Tp __d, memory_order = memory_order_seq_cst) noexcept {
    return std::__sp_atomic_compare_exchange_strong(std::addressof(__p), std::addressof(__e), __d);
  }

  // P1644R0 not implemented
  // void wait(_Tp old, memory_order order = memory_order::seq_cst) const noexcept;
  // void notify_one() noexcept;
  // void notify_all() noexcept;

private:
  _Tp __p;
};

template <class _Tp>
struct atomic<shared_ptr<_Tp>> : __sp_atomic_base<shared_ptr<_Tp>> {
  _LIBCPP_HIDE_FROM_ABI constexpr atomic() noexcept = default;
  _LIBCPP_HIDE_FROM_ABI constexpr atomic(nullptr_t) noexcept : atomic() {}
  _LIBCPP_HIDE_FROM_ABI atomic(shared_ptr<_Tp> desired) noexcept
      : __sp_atomic_base<shared_ptr<_Tp>>(std::move(desired)) {}
  _LIBCPP_HIDE_FROM_ABI atomic(const atomic&) = delete;

  _LIBCPP_HIDE_FROM_ABI void operator=(const atomic&) = delete;
  using __sp_atomic_base<shared_ptr<_Tp>>::operator=;
};

template <class _Tp>
struct atomic<weak_ptr<_Tp>> : __sp_atomic_base<weak_ptr<_Tp>> {
  _LIBCPP_HIDE_FROM_ABI constexpr atomic() noexcept = default;
  _LIBCPP_HIDE_FROM_ABI constexpr atomic(nullptr_t) noexcept : atomic() {}
  _LIBCPP_HIDE_FROM_ABI atomic(weak_ptr<_Tp> desired) noexcept : __sp_atomic_base<weak_ptr<_Tp>>(std::move(desired)) {}
  _LIBCPP_HIDE_FROM_ABI atomic(const atomic&) = delete;

  _LIBCPP_HIDE_FROM_ABI void operator=(const atomic&) = delete;
  using __sp_atomic_base<weak_ptr<_Tp>>::operator=;
};
#  endif // _LIBCPP_STD_VER >= 20

// [depr.util.smartptr.shared.atomic]

template <class _Tp>
inline _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_HIDE_FROM_ABI bool atomic_is_lock_free(const shared_ptr<_Tp>*) {
  return false;
}

template <class _Tp>
_LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_HIDE_FROM_ABI shared_ptr<_Tp>
atomic_load(const shared_ptr<_Tp>* __p) {
  return std::__sp_atomic_load(__p);
}

template <class _Tp>
inline _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_HIDE_FROM_ABI shared_ptr<_Tp>
atomic_load_explicit(const shared_ptr<_Tp>* __p, memory_order) {
  return std::atomic_load(__p);
}

template <class _Tp>
_LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_HIDE_FROM_ABI void atomic_store(shared_ptr<_Tp>* __p, shared_ptr<_Tp> __r) {
  std::__sp_atomic_store(__p, __r);
}

template <class _Tp>
inline _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_HIDE_FROM_ABI void
atomic_store_explicit(shared_ptr<_Tp>* __p, shared_ptr<_Tp> __r, memory_order) {
  std::atomic_store(__p, __r);
}

template <class _Tp>
_LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_HIDE_FROM_ABI shared_ptr<_Tp>
atomic_exchange(shared_ptr<_Tp>* __p, shared_ptr<_Tp> __r) {
  return std::__sp_atomic_exchange(__p, __r);
}

template <class _Tp>
inline _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_HIDE_FROM_ABI shared_ptr<_Tp>
atomic_exchange_explicit(shared_ptr<_Tp>* __p, shared_ptr<_Tp> __r, memory_order) {
  return std::atomic_exchange(__p, std::move(__r));
}

template <class _Tp>
_LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_HIDE_FROM_ABI bool
atomic_compare_exchange_strong(shared_ptr<_Tp>* __p, shared_ptr<_Tp>* __v, shared_ptr<_Tp> __w) {
  return std::__sp_atomic_compare_exchange_strong(__p, __v, __w);
}

template <class _Tp>
inline _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_HIDE_FROM_ABI bool
atomic_compare_exchange_weak(shared_ptr<_Tp>* __p, shared_ptr<_Tp>* __v, shared_ptr<_Tp> __w) {
  return std::atomic_compare_exchange_strong(__p, __v, std::move(__w));
}

template <class _Tp>
inline _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_HIDE_FROM_ABI bool atomic_compare_exchange_strong_explicit(
    shared_ptr<_Tp>* __p, shared_ptr<_Tp>* __v, shared_ptr<_Tp> __w, memory_order, memory_order) {
  return std::atomic_compare_exchange_strong(__p, __v, std::move(__w));
}

template <class _Tp>
inline _LIBCPP_DEPRECATED_IN_CXX20 _LIBCPP_HIDE_FROM_ABI bool atomic_compare_exchange_weak_explicit(
    shared_ptr<_Tp>* __p, shared_ptr<_Tp>* __v, shared_ptr<_Tp> __w, memory_order, memory_order) {
  return std::atomic_compare_exchange_weak(__p, __v, std::move(__w));
}

#endif // !defined(_LIBCPP_HAS_NO_THREADS)

_LIBCPP_END_NAMESPACE_STD

#endif // LLVM_ATOMIC_ATOMIC_SHARED_PTR_H
