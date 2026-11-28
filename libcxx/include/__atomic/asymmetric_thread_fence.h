// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ATOMIC_ASYMMETRIC_THREAD_FENCE_H
#define _LIBCPP___ATOMIC_ASYMMETRIC_THREAD_FENCE_H

#include <__atomic/atomic.h>
#include <__atomic/fence.h>
#include <__atomic/memory_order.h>
#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if !defined(_LIBCPP_CXX03_LANG) && _LIBCPP_HAS_THREADS

_LIBCPP_BEGIN_NAMESPACE_STD

// Asymmetric thread fences, shaped after Concurrency TS 2 (P1202) asymmetric_thread_fence_{light,heavy}.
// A light fence and a heavy fence order memory as if both were atomic_thread_fence(memory_order_seq_cst);
// two light fences order nothing with respect to each other. The light fence is meant for hot paths and
// the heavy fence for rare ones.
//
// Whether the heavy fence is a process-wide barrier (which lets the light fence be a compiler barrier) is
// decided by the built library, not by these headers: __asymmetric_thread_fence_is_asymmetric becomes true
// at most once, inside the first heavy fence, and only if every heavy fence from then on is such a barrier.
// Where _LIBCPP_HAS_GUARANTEED_ASYMMETRIC_THREAD_FENCE is set (Windows), the heavy fence always is one, so the
// light fence does not read the flag. Both fences currently ignore their memory_order argument and behave as
// seq_cst.
//
// Where the dylib may lack these symbols (vendor availability annotations, i.e. Apple back-deployment), the
// fences are unavailable, and so must be anything that uses them: there is no inline fallback. A fallback heavy
// fence would be compiled into user code and keep running on newer systems without reaching the library, so the
// library could never make the light fence a compiler barrier there.
//
// To support such deployment targets later, never pick a fallback at compile time. Check at runtime in both
// fences instead (__builtin_available, with weak imports of the symbols): the heavy fence calls the library when
// it has these symbols and is a plain fence otherwise; the light fence reads the flag only in the first case and
// is a full fence otherwise. Every piece of code in a process sees the same library, so all of it takes the
// same branch.

_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
extern _LIBCPP_AVAILABILITY_ASYMMETRIC_THREAD_FENCE _LIBCPP_EXPORTED_FROM_ABI atomic<bool>
    __asymmetric_thread_fence_is_asymmetric;
_LIBCPP_AVAILABILITY_ASYMMETRIC_THREAD_FENCE
_LIBCPP_EXPORTED_FROM_ABI void __asymmetric_thread_fence_heavy_impl(memory_order) noexcept;
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS

_LIBCPP_AVAILABILITY_ASYMMETRIC_THREAD_FENCE inline void
__asymmetric_thread_fence_light([[__maybe_unused__]] memory_order __order) noexcept {
#  if _LIBCPP_HAS_GUARANTEED_ASYMMETRIC_THREAD_FENCE
  std::atomic_signal_fence(memory_order_seq_cst);
#  else
  if (__asymmetric_thread_fence_is_asymmetric.load(memory_order_relaxed))
    std::atomic_signal_fence(memory_order_seq_cst);
  else
    std::atomic_thread_fence(memory_order_seq_cst);
#  endif
}

_LIBCPP_AVAILABILITY_ASYMMETRIC_THREAD_FENCE inline void
__asymmetric_thread_fence_heavy(memory_order __order) noexcept {
  std::__asymmetric_thread_fence_heavy_impl(__order);
}

_LIBCPP_END_NAMESPACE_STD

#endif // !defined(_LIBCPP_CXX03_LANG) && _LIBCPP_HAS_THREADS

#endif // _LIBCPP___ATOMIC_ASYMMETRIC_THREAD_FENCE_H
