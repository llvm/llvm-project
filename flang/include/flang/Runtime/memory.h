//===-- include/flang/Runtime/memory.h --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Thin wrapper around malloc()/free() to isolate the dependency,
// ease porting, and provide an owning pointer.

#ifndef FORTRAN_RUNTIME_MEMORY_H_
#define FORTRAN_RUNTIME_MEMORY_H_

#include "flang/Common/api-attrs.h"
#include <cassert>
#include <memory>
#include <type_traits>

namespace Fortran::runtime {

class Terminator;

[[nodiscard]] RT_API_ATTRS void *AllocateMemoryOrCrash(
    const Terminator &, std::size_t bytes);
template <typename A>
[[nodiscard]] RT_API_ATTRS A &AllocateOrCrash(const Terminator &t) {
  return *reinterpret_cast<A *>(AllocateMemoryOrCrash(t, sizeof(A)));
}
RT_API_ATTRS void *ReallocateMemoryOrCrash(
    const Terminator &, void *ptr, std::size_t newByteSize);
RT_API_ATTRS void FreeMemory(void *);
template <typename A> RT_API_ATTRS void FreeMemory(A *p) {
  FreeMemory(reinterpret_cast<void *>(p));
}
template <typename A> RT_API_ATTRS void FreeMemoryAndNullify(A *&p) {
  FreeMemory(p);
  p = nullptr;
}

// Very basic implementation mimicking std::unique_ptr.
// It should work for any offload device compiler.
// It uses a fixed memory deleter based on FreeMemory(),
// and does not support array objects with runtime length.
template <typename A> class OwningPtr {
public:
  using pointer_type = A *;

  OwningPtr() = default;
  RT_API_ATTRS explicit OwningPtr(pointer_type p) : ptr_(p) {}
  RT_API_ATTRS OwningPtr(const OwningPtr &) = delete;
  RT_API_ATTRS OwningPtr &operator=(const OwningPtr &) = delete;
  RT_API_ATTRS OwningPtr(OwningPtr &&other) {
    ptr_ = other.ptr_;
    other.ptr_ = pointer_type{};
  }
  RT_API_ATTRS OwningPtr &operator=(OwningPtr &&other) {
    if (this != &other) {
      delete_ptr(ptr_);
      ptr_ = other.ptr_;
      other.ptr_ = pointer_type{};
    }
    return *this;
  }
  constexpr RT_API_ATTRS OwningPtr(std::nullptr_t) : OwningPtr() {}

  // Delete the pointer, if owns one.
  RT_API_ATTRS ~OwningPtr() {
    if (ptr_ != pointer_type{}) {
      delete_ptr(ptr_);
      ptr_ = pointer_type{};
    }
  }

  // Release the ownership.
  RT_API_ATTRS pointer_type release() {
    pointer_type p = ptr_;
    ptr_ = pointer_type{};
    return p;
  }

  RT_DIAG_PUSH
  RT_DIAG_DISABLE_CALL_HOST_FROM_DEVICE_WARN
  // Replace the pointer.
  RT_API_ATTRS void reset(pointer_type p = pointer_type{}) {
    std::swap(ptr_, p);
    if (p != pointer_type{}) {
      // Delete the owned pointer.
      delete_ptr(p);
    }
  }

  // Exchange the pointer with another object.
  RT_API_ATTRS void swap(OwningPtr &other) { std::swap(ptr_, other.ptr_); }
  RT_DIAG_POP

  // Get the stored pointer.
  RT_API_ATTRS pointer_type get() const { return ptr_; }

  RT_API_ATTRS explicit operator bool() const {
    return get() != pointer_type{};
  }

  RT_API_ATTRS typename std::add_lvalue_reference<A>::type operator*() const {
    assert(get() != pointer_type{});
    return *get();
  }

  RT_API_ATTRS pointer_type operator->() const { return get(); }

private:
  RT_API_ATTRS void delete_ptr(pointer_type p) { FreeMemory(p); }
  pointer_type ptr_{};
};

template <typename X, typename Y>
inline RT_API_ATTRS bool operator!=(
    const OwningPtr<X> &x, const OwningPtr<Y> &y) {
  return x.get() != y.get();
}

template <typename X>
inline RT_API_ATTRS bool operator!=(const OwningPtr<X> &x, std::nullptr_t) {
  return (bool)x;
}

template <typename X>
inline RT_API_ATTRS bool operator!=(std::nullptr_t, const OwningPtr<X> &x) {
  return (bool)x;
}

template <typename A> class SizedNew {
public:
  explicit RT_API_ATTRS SizedNew(const Terminator &terminator)
      : terminator_{terminator} {}

  template <typename... X>
  [[nodiscard]] RT_API_ATTRS OwningPtr<A> operator()(
      std::size_t bytes, X &&...x) {
    return OwningPtr<A>{new (AllocateMemoryOrCrash(terminator_, bytes))
            A{std::forward<X>(x)...}};
  }

private:
  const Terminator &terminator_;
};

template <typename A> struct New : public SizedNew<A> {
  using SizedNew<A>::SizedNew;
  template <typename... X>
  [[nodiscard]] RT_API_ATTRS OwningPtr<A> operator()(X &&...x) {
    return SizedNew<A>::operator()(sizeof(A), std::forward<X>(x)...);
  }
};

template <typename A> struct Allocator {
  using value_type = A;
  explicit Allocator(const Terminator &t) : terminator{t} {}
  template <typename B>
  explicit constexpr Allocator(const Allocator<B> &that) noexcept
      : terminator{that.terminator} {}
  Allocator(const Allocator &) = default;
  Allocator(Allocator &&) = default;
  [[nodiscard]] constexpr A *allocate(std::size_t n) {
    return reinterpret_cast<A *>(
        AllocateMemoryOrCrash(terminator, n * sizeof(A)));
  }
  constexpr void deallocate(A *p, std::size_t) { FreeMemory(p); }
  const Terminator &terminator;
};
} // namespace Fortran::runtime

#endif // FORTRAN_RUNTIME_MEMORY_H_
