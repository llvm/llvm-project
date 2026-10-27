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

#if _LIBCPP_HAS_THREADS

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
// Where the dylib may lack these symbols (vendor availability annotations, e.g. Apple back-deployment), both
// fences are plain inline fences. Such a heavy fence never reaches the library, so on these platforms the
// library must keep the flag false (and the light fence a full fence) as long as such code may run in the
// process; otherwise a light fence from newer code could pair with an old inline heavy fence and order
// nothing. Today the only such platform is Apple, where the library's heavy fence is a plain fence anyway.
//
// TODO: once the LLVM 24 versions are filled in for these platforms, make the heavy fence check availability
// at runtime (__builtin_available) and call into the library when it has these symbols, so that code built
// from then on no longer inlines it. Code built before that keeps its inline heavy fence.

#  if _LIBCPP_AVAILABILITY_HAS_ASYMMETRIC_THREAD_FENCE
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
extern _LIBCPP_EXPORTED_FROM_ABI atomic<bool> __asymmetric_thread_fence_is_asymmetric;
_LIBCPP_EXPORTED_FROM_ABI void __asymmetric_thread_fence_heavy_impl(memory_order) _NOEXCEPT;
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
#  endif

_LIBCPP_HIDE_FROM_ABI inline void __asymmetric_thread_fence_light([[__maybe_unused__]] memory_order __order) _NOEXCEPT {
#  if !_LIBCPP_AVAILABILITY_HAS_ASYMMETRIC_THREAD_FENCE
  std::atomic_thread_fence(memory_order_seq_cst);
#  elif _LIBCPP_HAS_GUARANTEED_ASYMMETRIC_THREAD_FENCE
  std::atomic_signal_fence(memory_order_seq_cst);
#  else
  if (__asymmetric_thread_fence_is_asymmetric.load(memory_order_relaxed))
    std::atomic_signal_fence(memory_order_seq_cst);
  else
    std::atomic_thread_fence(memory_order_seq_cst);
#  endif
}

_LIBCPP_HIDE_FROM_ABI inline void __asymmetric_thread_fence_heavy([[__maybe_unused__]] memory_order __order) _NOEXCEPT {
#  if !_LIBCPP_AVAILABILITY_HAS_ASYMMETRIC_THREAD_FENCE
  std::atomic_thread_fence(memory_order_seq_cst);
#  else
  std::__asymmetric_thread_fence_heavy_impl(__order);
#  endif
}

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_HAS_THREADS

#endif // _LIBCPP___ATOMIC_ASYMMETRIC_THREAD_FENCE_H
