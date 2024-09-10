//===------------- Linux VDSO Header ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_VDSO_H
#define LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_VDSO_H
#include "hdr/types/clock_t.h"
#include "hdr/types/clockid_t.h"
#include "hdr/types/struct_timespec.h"
#include "hdr/types/struct_timeval.h"
#include "hdr/types/time_t.h"
#include "src/__support/CPP/array.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/properties/architectures.h"
#include "src/__support/threads/callonce.h"

// NOLINTBEGIN(llvmlibc-implementation-in-namespace)
// TODO: some of the following can be defined via proxy headers.
struct __kernel_timespec;
struct timezone;
struct riscv_hwprobe;
struct getcpu_cache;
struct cpu_set_t;
// NOLINTEND(llvmlibc-implementation-in-namespace)

namespace LIBC_NAMESPACE_DECL {
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
} // namespace LIBC_NAMESPACE_DECL

#if defined(LIBC_TARGET_ARCH_IS_X86)
#include "x86_64/vdso.h"
#elif defined(LIBC_TARGET_ARCH_IS_AARCH64)
#include "aarch64/vdso.h"
#elif defined(LIBC_TARGET_ARCH_IS_ARM)
#include "arm/vdso.h"
#elif defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
#include "riscv/vdso.h"
#else
#error "unknown arch"
#endif

namespace LIBC_NAMESPACE {
namespace vdso {

template <VDSOSym sym> LIBC_INLINE constexpr auto dispatcher() {
  if constexpr (sym == VDSOSym::ClockGetTime)
    return static_cast<int (*)(clockid_t, timespec *)>(nullptr);
  else if constexpr (sym == VDSOSym::ClockGetTime64)
    return static_cast<int (*)(clockid_t, __kernel_timespec *)>(nullptr);
  else if constexpr (sym == VDSOSym::GetTimeOfDay)
    return static_cast<int (*)(timeval *__restrict, timezone *__restrict)>(
        nullptr);
  else if constexpr (sym == VDSOSym::GetCpu)
    return static_cast<int (*)(unsigned *, unsigned *, getcpu_cache *)>(
        nullptr);
  else if constexpr (sym == VDSOSym::Time)
    return static_cast<time_t (*)(time_t *)>(nullptr);
  else if constexpr (sym == VDSOSym::ClockGetRes)
    return static_cast<int (*)(clockid_t, timespec *)>(nullptr);
  else if constexpr (sym == VDSOSym::RTSigReturn)
    return static_cast<void (*)(void)>(nullptr);
  else if constexpr (sym == VDSOSym::FlushICache)
    return static_cast<void (*)(void *, void *, unsigned int)>(nullptr);
  else if constexpr (sym == VDSOSym::RiscvHwProbe)
    return static_cast<int (*)(riscv_hwprobe *, size_t, size_t, cpu_set_t *,
                               unsigned)>(nullptr);
  else
    return static_cast<void *>(nullptr);
}

template <VDSOSym sym> using VDSOSymType = decltype(dispatcher<sym>());

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

  template <typename T> LIBC_INLINE T get() {
    if (name().empty() || !is_valid())
      return nullptr;

    callonce(&once_flag, Symbol::initialize_vdso_global_cache);
    return cpp::bit_cast<T>(global_cache[*this]);
  }
  template <VDSOSym sym> friend struct TypedSymbol;
};

template <VDSOSym sym> struct TypedSymbol {
  LIBC_INLINE constexpr operator VDSOSymType<sym>() const {
    return Symbol{sym}.get<VDSOSymType<sym>>();
  }
  template <typename... Args> LIBC_INLINE auto operator()(Args &&...args) {
    return this->operator VDSOSymType<sym>()(cpp::forward<Args>(args)...);
  }
};

} // namespace vdso

} // namespace LIBC_NAMESPACE
#endif // LLVM_LIBC_SRC___SUPPORT_OSUTIL_LINUX_VDSO_H
