//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Static TLS layout across loaded modules
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC___SUPPORT_ELF_TLS_LAYOUT_H
#define LLVM_LIBC_SRC___SUPPORT_ELF_TLS_LAYOUT_H

#include "hdr/link_macros.h"
#include "hdr/stdint_proxy.h"
#include "hdr/types/size_t.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"
#include "src/__support/macros/properties/architectures.h"

namespace LIBC_NAMESPACE_DECL {
namespace elf {

// Which side of the thread pointer the TLS blocks live on.
//
// Variant 2 puts them below the thread pointer, so a module's offset is
// subtracted from it. Variant 1 puts them above, after a two word thread
// control block. x86 uses variant 2 and everything else here uses variant 1.
#if defined(LIBC_TARGET_ARCH_IS_X86_64) || defined(LIBC_TARGET_ARCH_IS_X86_32)
constexpr bool TLS_VARIANT_2 = true;
#else
constexpr bool TLS_VARIANT_2 = false;
#endif

// Size of the thread control block that variant 1 reserves before the first
// module's block.
constexpr size_t TLS_TCB_SIZE = 2 * sizeof(void *);

// Bytes to keep at the thread pointer itself. Variant 1 needs only the
// control block, which its offsets already skip. Variant 2 has to leave room
// for what x86 addresses through the thread pointer by fixed offset: the self
// pointer at zero, the stack guard at 0x28 and the pointer guard at 0x30.
#if defined(LIBC_TARGET_ARCH_IS_X86_64) || defined(LIBC_TARGET_ARCH_IS_X86_32)
constexpr size_t TLS_TCB_RESERVE = 64;
#else
constexpr size_t TLS_TCB_RESERVE = TLS_TCB_SIZE;
#endif

// Where the stack guard sits, relative to the thread pointer, for code built
// with -fstack-protector.
constexpr size_t TLS_STACK_GUARD_OFFSET = 0x28;

LIBC_INLINE constexpr size_t round_up(size_t value, size_t alignment) {
  if (alignment <= 1)
    return value;
  return (value + alignment - 1) / alignment * alignment;
}

// Accumulates the static TLS layout as modules are added, in load order.
//
// This is the calculation a shared libc currently gets wrong: crt1.o sizes the
// thread block from the main executable's PT_TLS alone, so a libc.so with a
// PT_TLS segment of its own has nowhere to put it.
class TlsLayout {
public:
  LIBC_INLINE constexpr TlsLayout() = default;

  // Adds one module's PT_TLS and returns its offset from the thread pointer.
  // Under variant 2 the returned value is subtracted from the thread pointer;
  // under variant 1 it is added. A module with no TLS should not be added.
  LIBC_INLINE intptr_t add(size_t memsz, size_t alignment) {
    if (alignment == 0)
      alignment = 1;
    if (alignment > max_align_)
      max_align_ = alignment;

    if (TLS_VARIANT_2) {
      // Offsets grow downwards, so the running total is rounded up after the
      // new block is included.
      offset_ = round_up(offset_ + memsz, alignment);
      ++count_;
      return -static_cast<intptr_t>(offset_);
    }

    if (count_ == 0)
      offset_ = round_up(TLS_TCB_SIZE, alignment);
    else
      offset_ = round_up(offset_, alignment);
    const size_t here = offset_;
    offset_ += memsz;
    ++count_;
    return static_cast<intptr_t>(here);
  }

  // Total bytes the thread block needs, thread control block included.
  LIBC_INLINE constexpr size_t size() const {
    if (TLS_VARIANT_2)
      return offset_ + TLS_TCB_RESERVE;
    return offset_;
  }

  LIBC_INLINE constexpr size_t alignment() const { return max_align_; }
  LIBC_INLINE constexpr size_t module_count() const { return count_; }

private:
  size_t offset_ = 0;
  size_t max_align_ = TLS_TCB_SIZE;
  size_t count_ = 0;
};

} // namespace elf
} // namespace LIBC_NAMESPACE_DECL

#endif // LLVM_LIBC_SRC___SUPPORT_ELF_TLS_LAYOUT_H
