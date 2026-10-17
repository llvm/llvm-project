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
// On Windows the library always provides one. Both fences currently ignore their memory_order argument and
// behave as seq_cst.
//
// Where the dylib may lack these symbols (vendor availability annotations, e.g. Apple back-deployment), both
// fences are plain inline fences. A heavy fence compiled that way can never become a process-wide barrier, so
// on such platforms the library must keep the flag false (and the light fence a full fence) forever; otherwise
// a light fence from newer code could pair with an old inline heavy fence and order nothing.

#  if _LIBCPP_AVAILABILITY_HAS_ASYMMETRIC_THREAD_FENCE
_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS
extern _LIBCPP_EXPORTED_FROM_ABI atomic<bool> __asymmetric_thread_fence_is_asymmetric;
_LIBCPP_EXPORTED_FROM_ABI void __asymmetric_thread_fence_heavy_impl(memory_order) _NOEXCEPT;
_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS
#  endif

_LIBCPP_HIDE_FROM_ABI inline void __asymmetric_thread_fence_light([[__maybe_unused__]] memory_order __order) _NOEXCEPT {
#  if !_LIBCPP_AVAILABILITY_HAS_ASYMMETRIC_THREAD_FENCE
  std::atomic_thread_fence(memory_order_seq_cst);
#  elif defined(_WIN32)
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
