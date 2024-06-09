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

namespace LIBC_NAMESPACE {
namespace vdso {

#ifdef __clang__
__extension__ template <typename T> using NullablePtr = T *_Nullable;
__extension__ template <typename T> using NonNullPtr = T *_Nonnull;
#else
template <typename T> using NullablePtr = T *;
template <typename T> using NonNullPtr = T *;
#endif

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
#elif defined(LIBC_TARGET_ARCH_IS_ANY_RISCV)
#include "riscv/vdso.h"
#else
#error "unknown arch"
#endif

namespace LIBC_NAMESPACE {
namespace vdso {

template <VDSOSym sym> struct VDSOTypeDispatch {
  using Type = void *;
};

template <> struct VDSOTypeDispatch<VDSOSym::ClockGetTime> {
  using Type = int (*)(clockid_t, NonNullPtr<timespec>);
};

template <> struct VDSOTypeDispatch<VDSOSym::ClockGetTime64> {
  using Type = int (*)(clockid_t, NonNullPtr<__kernel_timespec>);
};

template <> struct VDSOTypeDispatch<VDSOSym::GetTimeOfDay> {
  using Type = int (*)(NonNullPtr<timeval> __restrict,
                       NullablePtr<timezone> __restrict);
};

template <> struct VDSOTypeDispatch<VDSOSym::GetCpu> {
  using Type = int (*)(NullablePtr<unsigned>, NullablePtr<unsigned>,
                       NullablePtr<getcpu_cache>);
};

template <> struct VDSOTypeDispatch<VDSOSym::Time> {
  using Type = time_t (*)(NullablePtr<time_t>);
};

template <> struct VDSOTypeDispatch<VDSOSym::ClockGetRes> {
  using Type = int (*)(clockid_t, NullablePtr<timespec>);
};

template <> struct VDSOTypeDispatch<VDSOSym::RTSigReturn> {
  using Type = void (*)(void);
};

template <> struct VDSOTypeDispatch<VDSOSym::FlushICache> {
  using Type = void (*)(NullablePtr<void>, NullablePtr<void>, unsigned int);
};

template <> struct VDSOTypeDispatch<VDSOSym::RiscvHwProbe> {
  using Type = int (*)(NullablePtr<riscv_hwprobe> __restrict, size_t, size_t,
                       NullablePtr<cpu_set_t> __restrict, unsigned);
};

template <VDSOSym sym> using VDSOSymType = typename VDSOTypeDispatch<sym>::Type;

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
