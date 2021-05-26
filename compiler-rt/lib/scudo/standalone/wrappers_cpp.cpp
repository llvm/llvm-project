//===-- wrappers_cpp.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "platform.h"

// Skip this compilation unit if compiled as part of Bionic.
#if !SCUDO_ANDROID || !_BIONIC

#include "allocator_config.h"

#include <stdint.h>

extern "C" void malloc_postinit();
extern HIDDEN scudo::Allocator<scudo::Config, malloc_postinit> Allocator;

namespace std {
struct nothrow_t {};
enum class align_val_t : size_t {};
} // namespace std

// It's not valid for a non-throwing non-noreturn operator new to return
// nullptr. In these cases, just terminate.
#define CHECK_ALIGN_NORETURN(align)                                            \
  do {                                                                         \
    scudo::uptr alignment = static_cast<scudo::uptr>(align);                   \
    if (UNLIKELY(!scudo::isPowerOfTwo(alignment))) {                           \
      scudo::reportAlignmentNotPowerOfTwo(alignment);                          \
    }                                                                          \
  } while (0)

#define CHECK_ALIGN(align)                                                     \
  do {                                                                         \
    scudo::uptr alignment = static_cast<scudo::uptr>(align);                   \
    if (UNLIKELY(!scudo::isPowerOfTwo(alignment))) {                           \
      if (Allocator.canReturnNull()) {                                         \
        return nullptr;                                                        \
      }                                                                        \
      scudo::reportAlignmentNotPowerOfTwo(alignment);                          \
    }                                                                          \
  } while (0)

INTERFACE WEAK void *operator new(size_t size) {
  return Allocator.allocate(size, scudo::Chunk::Origin::New);
}
INTERFACE WEAK void *operator new[](size_t size) {
  return Allocator.allocate(size, scudo::Chunk::Origin::NewArray);
}
INTERFACE WEAK void *operator new(size_t size,
                                  std::nothrow_t const &) NOEXCEPT {
  return Allocator.allocate(size, scudo::Chunk::Origin::New);
}
INTERFACE WEAK void *operator new[](size_t size,
                                    std::nothrow_t const &) NOEXCEPT {
  return Allocator.allocate(size, scudo::Chunk::Origin::NewArray);
}
INTERFACE WEAK void *operator new(size_t size, std::align_val_t align) {
  CHECK_ALIGN_NORETURN(align);
  return Allocator.allocate(size, scudo::Chunk::Origin::New,
                            static_cast<scudo::uptr>(align));
}
INTERFACE WEAK void *operator new[](size_t size, std::align_val_t align) {
  CHECK_ALIGN_NORETURN(align);
  return Allocator.allocate(size, scudo::Chunk::Origin::NewArray,
                            static_cast<scudo::uptr>(align));
}
INTERFACE WEAK void *operator new(size_t size, std::align_val_t align,
                                  std::nothrow_t const &) NOEXCEPT {
  CHECK_ALIGN(align);
  return Allocator.allocate(size, scudo::Chunk::Origin::New,
                            static_cast<scudo::uptr>(align));
}
INTERFACE WEAK void *operator new[](size_t size, std::align_val_t align,
                                    std::nothrow_t const &) NOEXCEPT {
  CHECK_ALIGN(align);
  return Allocator.allocate(size, scudo::Chunk::Origin::NewArray,
                            static_cast<scudo::uptr>(align));
}

INTERFACE WEAK void operator delete(void *ptr)NOEXCEPT {
  Allocator.deallocate(ptr, scudo::Chunk::Origin::New);
}
INTERFACE WEAK void operator delete[](void *ptr) NOEXCEPT {
  Allocator.deallocate(ptr, scudo::Chunk::Origin::NewArray);
}
INTERFACE WEAK void operator delete(void *ptr, std::nothrow_t const &)NOEXCEPT {
  Allocator.deallocate(ptr, scudo::Chunk::Origin::New);
}
INTERFACE WEAK void operator delete[](void *ptr,
                                      std::nothrow_t const &) NOEXCEPT {
  Allocator.deallocate(ptr, scudo::Chunk::Origin::NewArray);
}
INTERFACE WEAK void operator delete(void *ptr, size_t size)NOEXCEPT {
  Allocator.deallocate(ptr, scudo::Chunk::Origin::New, size);
}
INTERFACE WEAK void operator delete[](void *ptr, size_t size) NOEXCEPT {
  Allocator.deallocate(ptr, scudo::Chunk::Origin::NewArray, size);
}
INTERFACE WEAK void operator delete(void *ptr, std::align_val_t align)NOEXCEPT {
  Allocator.deallocate(ptr, scudo::Chunk::Origin::New, 0,
                       static_cast<scudo::uptr>(align));
}
INTERFACE WEAK void operator delete[](void *ptr,
                                      std::align_val_t align) NOEXCEPT {
  Allocator.deallocate(ptr, scudo::Chunk::Origin::NewArray, 0,
                       static_cast<scudo::uptr>(align));
}
INTERFACE WEAK void operator delete(void *ptr, std::align_val_t align,
                                    std::nothrow_t const &)NOEXCEPT {
  Allocator.deallocate(ptr, scudo::Chunk::Origin::New, 0,
                       static_cast<scudo::uptr>(align));
}
INTERFACE WEAK void operator delete[](void *ptr, std::align_val_t align,
                                      std::nothrow_t const &) NOEXCEPT {
  Allocator.deallocate(ptr, scudo::Chunk::Origin::NewArray, 0,
                       static_cast<scudo::uptr>(align));
}
INTERFACE WEAK void operator delete(void *ptr, size_t size,
                                    std::align_val_t align)NOEXCEPT {
  Allocator.deallocate(ptr, scudo::Chunk::Origin::New, size,
                       static_cast<scudo::uptr>(align));
}
INTERFACE WEAK void operator delete[](void *ptr, size_t size,
                                      std::align_val_t align) NOEXCEPT {
  Allocator.deallocate(ptr, scudo::Chunk::Origin::NewArray, size,
                       static_cast<scudo::uptr>(align));
}

#endif // !SCUDO_ANDROID || !_BIONIC
