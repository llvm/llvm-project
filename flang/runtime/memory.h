//===-- runtime/memory.h ----------------------------------------*- C++ -*-===//
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

#include <memory>

namespace Fortran::runtime {

class Terminator;

[[nodiscard]] void *AllocateMemoryOrCrash(
    const Terminator &, std::size_t bytes);
template <typename A> [[nodiscard]] A &AllocateOrCrash(const Terminator &t) {
  return *reinterpret_cast<A *>(AllocateMemoryOrCrash(t, sizeof(A)));
}
void FreeMemory(void *);
template <typename A> void FreeMemory(A *p) {
  FreeMemory(reinterpret_cast<void *>(p));
}
template <typename A> void FreeMemoryAndNullify(A *&p) {
  FreeMemory(p);
  p = nullptr;
}

template <typename A> struct OwningPtrDeleter {
  void operator()(A *p) { FreeMemory(p); }
};

template <typename A> using OwningPtr = std::unique_ptr<A, OwningPtrDeleter<A>>;

template <typename A> class SizedNew {
public:
  explicit SizedNew(const Terminator &terminator) : terminator_{terminator} {}
  template <typename... X>
  [[nodiscard]] OwningPtr<A> operator()(std::size_t bytes, X &&...x) {
    return OwningPtr<A>{new (AllocateMemoryOrCrash(terminator_, bytes))
            A{std::forward<X>(x)...}};
  }

private:
  const Terminator &terminator_;
};

template <typename A> struct New : public SizedNew<A> {
  using SizedNew<A>::SizedNew;
  template <typename... X> [[nodiscard]] OwningPtr<A> operator()(X &&...x) {
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
