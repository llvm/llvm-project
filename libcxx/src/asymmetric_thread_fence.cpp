//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <__atomic/asymmetric_thread_fence.h>
#include <__atomic/fence.h>
#include <__atomic/memory_order.h>
#include <__config>
#include <__verbose_abort>

#if defined(__linux__)
#  include <cerrno>
#  include <sys/syscall.h>
#  include <unistd.h>
#elif defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  define VC_EXTRA_LEAN
#  include <windows.h>
#endif

_LIBCPP_BEGIN_NAMESPACE_STD
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

// See <__atomic/asymmetric_thread_fence.h> for the protocol. The flag may only become true once every heavy
// fence that can still run is a process-wide barrier, and it never becomes false again.
#if defined(_WIN32)
constinit atomic<bool> __asymmetric_thread_fence_is_asymmetric{true};
#else
constinit atomic<bool> __asymmetric_thread_fence_is_asymmetric{false};
#endif

#if defined(__linux__)

namespace {

#  if defined(SYS_membarrier)
// membarrier(2) commands. The values are kernel ABI; spelling them here keeps the build independent of the
// UAPI headers it was configured with.
constexpr int __membarrier_cmd_query                      = 0;
constexpr int __membarrier_cmd_private_expedited          = 1 << 3;
constexpr int __membarrier_cmd_register_private_expedited = 1 << 4;

long __membarrier(int __cmd) noexcept { return ::syscall(SYS_membarrier, __cmd, 0, 0); }

bool __enable_membarrier() noexcept {
  long __commands = __membarrier(__membarrier_cmd_query);
  return __commands != -1 && (__commands & __membarrier_cmd_private_expedited) != 0 &&
         __membarrier(__membarrier_cmd_register_private_expedited) == 0;
}
#  else
bool __enable_membarrier() noexcept { return false; }
#  endif

} // namespace

// TODO: add an mprotect()-based fallback (as folly does; possibly x86-only, since on AArch64 the TLB
// invalidation is broadcast in hardware and does not interrupt the other CPUs) for kernels without
// membarrier(2) PRIVATE_EXPEDITED. This is private to the library, so it can change without ABI impact.
void __asymmetric_thread_fence_heavy_impl([[maybe_unused]] memory_order __order) noexcept {
  // Decided once, before any heavy fence could be a plain fence while the flag reads true: concurrent first
  // calls wait for this initialization, and the flag is only ever set here.
  static const bool __asymmetric = [] {
    bool __ok = __enable_membarrier();
    if (__ok)
      __asymmetric_thread_fence_is_asymmetric.store(true, memory_order_relaxed);
    return __ok;
  }();
  if (!__asymmetric) {
    std::atomic_thread_fence(memory_order_seq_cst);
    return;
  }
#  if defined(SYS_membarrier)
  if (__membarrier(__membarrier_cmd_private_expedited) == 0)
    return;
  // Light fences may already rely on this barrier, so never fall back to a plain fence. The registration
  // survives fork() and only exec() clears it (which also reloads this library), so EPERM is not expected;
  // should the registration ever be lost anyway, register again.
  if (errno == EPERM && __membarrier(__membarrier_cmd_register_private_expedited) == 0 &&
      __membarrier(__membarrier_cmd_private_expedited) == 0)
    return;
#  endif
  __libcpp_verbose_abort("membarrier(2) failed after the asymmetric thread fence was enabled");
}

#else

void __asymmetric_thread_fence_heavy_impl([[maybe_unused]] memory_order __order) noexcept {
#  if defined(_WIN32)
  ::FlushProcessWriteBuffers();
#  else
  // TODO: process-wide barriers on more platforms (FreeBSD 14.1+ has membarrier(2)). Not on platforms with
  // vendor availability annotations (Apple): code back-deployed to a dylib without these symbols issues inline
  // plain heavy fences, so the flag must stay false there (see <__atomic/asymmetric_thread_fence.h>).
  std::atomic_thread_fence(memory_order_seq_cst);
#  endif
}

#endif

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
_LIBCPP_END_NAMESPACE_STD
