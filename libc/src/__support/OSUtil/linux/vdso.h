//===------------- Linux VDSO Header ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_VDSO_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_VDSO_H
#include "src/__support/CPP/array.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/threads/callonce.h"

namespace LIBC_NAMESPACE {
namespace vdso {
enum class VDSOSym {
  ClockGetTime,
  ClockGetTime64,
  GetTimeOfDay,
  GetCpu,
  Time,
  ClockGetRes,
  RTSigReturn,
  FlushICache,
  RiscvHwProbe,
  VDSOSymCount
};
} // namespace vdso
} // namespace LIBC_NAMESPACE

#if defined(LIBC_TARGET_ARCH_IS_X86)
#include "x86_64/vdso.h"
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "aarch64/vdso.h"
#elif defined(LIBC_TARGET_ARCH_IS_ARM)
#include "arm/vdso.h"
#elif defined(LIBC_TARGET_ARCH_IS_RISCV)
#include "riscv/vdso.h"
#else
#error "unknown arch"
#endif

namespace LIBC_NAMESPACE {
namespace vdso {
class Symbol {
  VDSOSym sym;

public:
  LIBC_INLINE_VAR static constexpr size_t COUNT =
      static_cast<size_t>(VDSOSym::VDSOSymCount);
  LIBC_INLINE constexpr explicit Symbol(VDSOSym sym) : sym(sym) {}
  LIBC_INLINE constexpr Symbol(size_t idx) : sym(static_cast<VDSOSym>(idx)) {}
  LIBC_INLINE constexpr cpp::string_view name() const {
    return symbol_name(sym);
  }
  LIBC_INLINE constexpr cpp::string_view version() const {
    return symbol_version(sym);
  }
  LIBC_INLINE constexpr operator size_t() const {
    return static_cast<size_t>(sym);
  }
  LIBC_INLINE constexpr bool is_valid() const {
    return *this < Symbol::global_cache.size();
  }
  using VDSOArray = cpp::array<void *, Symbol::COUNT>;

private:
  static CallOnceFlag once_flag;
  static VDSOArray global_cache;
  static void initialize_vdso_global_cache();

public:
  template <typename T> LIBC_INLINE T get() {
    if (name().empty() || !is_valid())
      return nullptr;

    callonce(&once_flag, Symbol::initialize_vdso_global_cache);
    return cpp::bit_cast<T>(global_cache[*this]);
  }
};

} // namespace vdso

} // namespace LIBC_NAMESPACE
#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_VDSO_H
