//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__memory/allocate_at_least.h>
#include <__new/new_at_least.h>
#include <cstddef>
#include <cstdlib>
#include <new>

#include "include/aligned_alloc.h"
#include "include/overridable_function.h"

// libc++ and libc++abi have different assertion mechanisms, so we need different implementations of
// `__throw_bad_alloc_shim` and `_LIBCPP_ASSERT_SHIM` for them in -fno-exceptions mode. These are expected to be
// provided before including this file.
void __throw_bad_alloc_shim();

#ifndef _LIBCPP_ASSERT_SHIM
#  error _LIBCPP_ASSERT_SHIM should be defined
#  define _LIBCPP_ASSERT_SHIM // make the file parseable
#endif

enum class on_failure {
  return_null,
  throw_bad_alloc,
};

template <on_failure failure_mode>
static void* operator_new_impl(std::size_t size) {
  if (size == 0)
    size = 1;
  void* p;
  while ((p = std::malloc(size)) == nullptr) {
    // If malloc fails and there is a new_handler,
    // call it to try free up memory.
    std::new_handler nh = std::get_new_handler();
    if (nh)
      nh();
    else
      break;
  }
  if (failure_mode == on_failure::throw_bad_alloc && !p)
    __throw_bad_alloc_shim();
  return p;
}

OVERRIDABLE_FUNCTION void* operator new(std::size_t size) _THROW_BAD_ALLOC {
  return operator_new_impl<on_failure::throw_bad_alloc>(size);
}

[[gnu::weak]] void* operator new(size_t size, const std::nothrow_t&) noexcept {
#if !_LIBCPP_HAS_EXCEPTIONS
#  if _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION
  _LIBCPP_ASSERT_SHIM(
      (!std::__is_function_overridden < void*(std::size_t), &operator new>()),
      "libc++ was configured with exceptions disabled and `operator new(size_t)` has been overridden, "
      "but `operator new(size_t, nothrow_t)` has not been overridden. This is problematic because "
      "`operator new(size_t, nothrow_t)` must call `operator new(size_t)`, which will terminate in case "
      "it fails to allocate, making it impossible for `operator new(size_t, nothrow_t)` to fulfill its "
      "contract (since it should return nullptr upon failure). Please make sure you override "
      "`operator new(size_t, nothrow_t)` as well.");
#  endif

  return operator_new_impl<on_failure::return_null>(size);
#else
  void* p = nullptr;
  try {
    p = ::operator new(size);
  } catch (...) {
  }
  return p;
#endif
}

OVERRIDABLE_FUNCTION void* operator new[](size_t size) _THROW_BAD_ALLOC { return ::operator new(size); }

[[gnu::weak]] void* operator new[](size_t size, const std::nothrow_t&) noexcept {
#if !_LIBCPP_HAS_EXCEPTIONS
#  if _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION
  _LIBCPP_ASSERT_SHIM(
      (!std::__is_function_overridden < void*(std::size_t), &operator new[]>()),
      "libc++ was configured with exceptions disabled and `operator new[](size_t)` has been overridden, "
      "but `operator new[](size_t, nothrow_t)` has not been overridden. This is problematic because "
      "`operator new[](size_t, nothrow_t)` must call `operator new[](size_t)`, which will terminate in case "
      "it fails to allocate, making it impossible for `operator new[](size_t, nothrow_t)` to fulfill its "
      "contract (since it should return nullptr upon failure). Please make sure you override "
      "`operator new[](size_t, nothrow_t)` as well.");
#  endif

  return operator_new_impl<on_failure::return_null>(size);
#else
  void* p = nullptr;
  try {
    p = ::operator new[](size);
  } catch (...) {
  }
  return p;
#endif
}

[[gnu::weak]] void operator delete(void* ptr) noexcept { std::free(ptr); }

[[gnu::weak]] void operator delete(void* ptr, const std::nothrow_t&) noexcept { ::operator delete(ptr); }

[[gnu::weak]] void operator delete(void* ptr, size_t) noexcept { ::operator delete(ptr); }

[[gnu::weak]] void operator delete[](void* ptr) noexcept { ::operator delete(ptr); }

[[gnu::weak]] void operator delete[](void* ptr, const std::nothrow_t&) noexcept { ::operator delete[](ptr); }

[[gnu::weak]] void operator delete[](void* ptr, size_t) noexcept { ::operator delete[](ptr); }

#if _LIBCPP_HAS_LIBRARY_ALIGNED_ALLOCATION

template <on_failure failure_mode>
static void* operator_new_aligned_impl(std::size_t size, std::align_val_t alignment) {
  if (size == 0)
    size = 1;
  if (static_cast<size_t>(alignment) < sizeof(void*))
    alignment = std::align_val_t(sizeof(void*));

  // Try allocating memory. If allocation fails and there is a new_handler,
  // call it to try free up memory, and try again until it succeeds, or until
  // the new_handler decides to terminate.
  void* p;
  while ((p = std::__libcpp_aligned_alloc(static_cast<std::size_t>(alignment), size)) == nullptr) {
    std::new_handler nh = std::get_new_handler();
    if (nh)
      nh();
    else
      break;
  }
  if (failure_mode == on_failure::throw_bad_alloc && !p)
    __throw_bad_alloc_shim();
  return p;
}

OVERRIDABLE_FUNCTION void* operator new(std::size_t size, std::align_val_t alignment) _THROW_BAD_ALLOC {
  return operator_new_aligned_impl<on_failure::throw_bad_alloc>(size, alignment);
}

[[gnu::weak]] void* operator new(size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
#  if !_LIBCPP_HAS_EXCEPTIONS
#    if _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION
  _LIBCPP_ASSERT_SHIM(
      (!std::__is_function_overridden < void*(std::size_t, std::align_val_t), &operator new>()),
      "libc++ was configured with exceptions disabled and `operator new(size_t, align_val_t)` has been overridden, "
      "but `operator new(size_t, align_val_t, nothrow_t)` has not been overridden. This is problematic because "
      "`operator new(size_t, align_val_t, nothrow_t)` must call `operator new(size_t, align_val_t)`, which will "
      "terminate in case it fails to allocate, making it impossible for `operator new(size_t, align_val_t, nothrow_t)` "
      "to fulfill its contract (since it should return nullptr upon failure). Please make sure you override "
      "`operator new(size_t, align_val_t, nothrow_t)` as well.");
#    endif

  return operator_new_aligned_impl<on_failure::return_null>(size, alignment);
#  else
  void* p = nullptr;
  try {
    p = ::operator new(size, alignment);
  } catch (...) {
  }
  return p;
#  endif
}

OVERRIDABLE_FUNCTION void* operator new[](size_t size, std::align_val_t alignment) _THROW_BAD_ALLOC {
  return ::operator new(size, alignment);
}

[[gnu::weak]] void* operator new[](size_t size, std::align_val_t alignment, const std::nothrow_t&) noexcept {
#  if !_LIBCPP_HAS_EXCEPTIONS
#    if _LIBCPP_CAN_DETECT_OVERRIDDEN_FUNCTION
  _LIBCPP_ASSERT_SHIM(
      (!std::__is_function_overridden < void*(std::size_t, std::align_val_t), &operator new[]>()),
      "libc++ was configured with exceptions disabled and `operator new[](size_t, align_val_t)` has been overridden, "
      "but `operator new[](size_t, align_val_t, nothrow_t)` has not been overridden. This is problematic because "
      "`operator new[](size_t, align_val_t, nothrow_t)` must call `operator new[](size_t, align_val_t)`, which will "
      "terminate in case it fails to allocate, making it impossible for `operator new[](size_t, align_val_t, "
      "nothrow_t)` to fulfill its contract (since it should return nullptr upon failure). Please make sure you "
      "override `operator new[](size_t, align_val_t, nothrow_t)` as well.");
#    endif

  return operator_new_aligned_impl<on_failure::return_null>(size, alignment);
#  else
  void* p = nullptr;
  try {
    p = ::operator new[](size, alignment);
  } catch (...) {
  }
  return p;
#  endif
}

[[gnu::weak]] void operator delete(void* ptr, std::align_val_t) noexcept { std::__libcpp_aligned_free(ptr); }

[[gnu::weak]] void operator delete(void* ptr, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  ::operator delete(ptr, alignment);
}

[[gnu::weak]] void operator delete(void* ptr, size_t, std::align_val_t alignment) noexcept {
  ::operator delete(ptr, alignment);
}

[[gnu::weak]] void operator delete[](void* ptr, std::align_val_t alignment) noexcept {
  ::operator delete(ptr, alignment);
}

[[gnu::weak]] void operator delete[](void* ptr, std::align_val_t alignment, const std::nothrow_t&) noexcept {
  ::operator delete[](ptr, alignment);
}

[[gnu::weak]] void operator delete[](void* ptr, size_t, std::align_val_t alignment) noexcept {
  ::operator delete[](ptr, alignment);
}
#endif // _LIBCPP_HAS_LIBRARY_ALIGNED_ALLOCATION

// This part implements __new_at_least, a version of operator new that returns the actually allocated amount of memory
// in addition to the pointer. Since users are allowed to replace operator new, we have to check whether it is replaced
// and fall back to that. Otherwise we can use platform-specific APIs for an improved implementation.
//
// We do that check via `gnu::ifunc` if it's available. Otherwise we don't do anything and just unconditionally forward
// to `operator new(size_t{, align_val_t})`. `gnu::ifunc` takes the mangled name of a resolver function. That resolver
// function returns a pointer to the function that should be linked. We make use of `__is_function_overridden` to detect
// whether we can use our own special implementation or have to fall back to a user-provided operator new. This approach
// avoids any repeated checks in a very hot path.

#ifdef __APPLE__
#  include <malloc/malloc.h>
#elifdef __FreeBSD__
#  include <malloc_np.h>
#endif

// FIXME: Clang should really accept functions in [[gnu::ifunc]] (or possibly [[clang::ifunc]])
using std::__allocation_result;

using new_t         = void*(std::size_t);
using new_aligned_t = void*(std::size_t, std::align_val_t);

using new_at_least_t         = __allocation_result<void*>(std::size_t);
using new_at_least_aligned_t = __allocation_result<void*>(std::size_t, std::align_val_t);

[[maybe_unused]] static new_at_least_t* new_at_least_resolver() {
  if (std::__is_function_overridden < new_t, operator new>()) {
    return [](std::size_t size) -> __allocation_result<void*> { return {::operator new(size), size}; };
  } else {
    return [](std::size_t size) -> __allocation_result<void*> {
#ifdef __APPLE__
      auto good_size = ::malloc_good_size(size);
      return {operator_new_impl<on_failure::throw_bad_alloc>(good_size), good_size};
#elifdef __FreeBSD__
      auto good_size = ::nallocx(size, 0);
      return {operator_new_impl<on_failure::throw_bad_alloc>(good_size), good_size};
#else
      // Other platforms should specialize this for their system allocator
      return {operator_new_impl<on_failure::throw_bad_alloc>(size), size};
#endif
    };
  }
}

[[maybe_unused]] static new_at_least_aligned_t* new_at_least_aligned_resolver() {
  if (std::__is_function_overridden < new_aligned_t, operator new>()) {
    return [](std::size_t size, std::align_val_t align) -> __allocation_result<void*> {
      return {::operator new(size, align), size};
    };
  } else {
    return [](std::size_t size, std::align_val_t align) -> __allocation_result<void*> {
#ifdef __APPLE__
      auto good_size = ::malloc_good_size(size);
      return {operator_new_aligned_impl<on_failure::throw_bad_alloc>(good_size, align), good_size};
#elifdef __FreeBSD__
      auto good_size = ::nallocx(size, MALLOCX_ALIGN(static_cast<size_t>(align)));
      return {operator_new_aligned_impl<on_failure::throw_bad_alloc>(good_size), good_size};
#else
      return {operator_new_aligned_impl<on_failure::throw_bad_alloc>(size, align), size};
#endif
    };
  }
}

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

#if __has_cpp_attribute(gnu::ifunc)

[[gnu::ifunc("_ZL21new_at_least_resolverv")]] new_at_least_t __new_at_least;
[[gnu::ifunc("_ZL29new_at_least_aligned_resolverv")]] new_at_least_aligned_t __new_at_least;

#else

std::__allocation_result<void*> __new_at_least(std::size_t size) { return {::operator new(size), size}; }

#  if _LIBCPP_HAS_LIBRARY_ALIGNED_ALLOCATION
std::__allocation_result<void*> __new_at_least(std::size_t size, std::align_val_t align) {
  return {::operator new(size, align), size};
}
#  endif

#endif

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
