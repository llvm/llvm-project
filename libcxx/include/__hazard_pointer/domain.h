// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___HAZARD_POINTER_DOMAIN_H
#define _LIBCPP___HAZARD_POINTER_DOMAIN_H

#include <__atomic/atomic.h>
#include <__atomic/fence.h>
#include <__atomic/memory_order.h>
#include <__config>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS

_LIBCPP_BEGIN_NAMESPACE_STD

// ABI contract between <hazard_pointer> and the built library (frozen since LLVM 24)
//
// - __hazard_pointer_slot is the first base of every hazard-pointer record owned by the library. It is
//   the only part of a record the headers know: try_protect() and reset_protection() store into
//   __value_ inline. sizeof(__hazard_pointer_slot) == sizeof(void*).
// - __hazard_pointer_obj_node is the private base of hazard_pointer_obj_base<T, D>. Retired lists are
//   intrusive, so the node lives inside every hazard-protectable object and its layout is part of the
//   user's object layout: {__next_, __reclaim_}, 2 * sizeof(void*). __next_ == this means "not retired".
// - __hazard_pointer_acquire(), __hazard_pointer_release() and __hazard_pointer_retire() are the only
//   entry points. acquire() returns an unassociated slot (__value_ == nullptr) which stays valid for the
//   whole process lifetime (records are never freed). release() ends the epoch itself (release-store of
//   nullptr) and returns the slot to the pool; the headers never touch a slot after handing it back.
//   retire() takes a node whose __reclaim_ is set, issues the retire-side fence, publishes the node and
//   may run reclamation (and thus deleters) synchronously on the calling thread. Deleters are never
//   invoked from acquire()/release() and never on a background thread. All three entry points are
//   callable during static destruction and thread teardown.
// - Reader protocol (v1, inline in the headers): reset_protection(p) is a release-store of the address
//   of p's node into __value_; try_protect() is: store, __hazard_pointer_reader_fence(), acquire-reload
//   of src, compare. The library loads slots with acquire ordering (or relaxed followed by an acquire
//   fence) and, before scanning them, issues a fence that orders every reader's store-then-load pair
//   against the scan: where _LIBCPP_HAZARD_POINTER_ASYMMETRIC_FENCE is 1 (Linux, Windows) that is a
//   process-wide barrier (membarrier(2) / FlushProcessWriteBuffers) and the reader's fence is a compiler
//   barrier; elsewhere both sides issue a seq_cst fence. The platform split is part of this contract: a
//   library built for one of those platforms must keep issuing the process-wide barrier, since headers
//   compiled against it emit no hardware fence on the reader side.

struct __hazard_pointer_slot {
  atomic<const void*> __value_;
};

struct __hazard_pointer_obj_node {
  using __reclaim_fn _LIBCPP_NODEBUG = void (*)(__hazard_pointer_obj_node*) noexcept;

  __hazard_pointer_obj_node* __next_;
  __reclaim_fn __reclaim_;

  // Every construction yields a fresh, not-retired node and assignment never touches the node: a copy
  // of a retired-but-still-protected object must itself be a fresh object.
  _LIBCPP_HIDE_FROM_ABI constexpr __hazard_pointer_obj_node() noexcept : __next_(this), __reclaim_(nullptr) {}
  _LIBCPP_HIDE_FROM_ABI constexpr __hazard_pointer_obj_node(const __hazard_pointer_obj_node&) noexcept
      : __next_(this), __reclaim_(nullptr) {}
  _LIBCPP_HIDE_FROM_ABI constexpr __hazard_pointer_obj_node& operator=(const __hazard_pointer_obj_node&) noexcept {
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI ~__hazard_pointer_obj_node() = default;
};

// Asymmetric fences (the reader side is a compiler barrier, the reclaimer issues a process-wide barrier)
// on the platforms where the built library has such a barrier; see the protocol note above.
#  if defined(__linux__) || defined(_WIN32)
#    define _LIBCPP_HAZARD_POINTER_ASYMMETRIC_FENCE 1
#  else
#    define _LIBCPP_HAZARD_POINTER_ASYMMETRIC_FENCE 0
#  endif

// The reader-side fence of try_protect().
_LIBCPP_HIDE_FROM_ABI inline void __hazard_pointer_reader_fence() noexcept {
#  if _LIBCPP_HAZARD_POINTER_ASYMMETRIC_FENCE
  std::atomic_signal_fence(memory_order_seq_cst);
#  else
  std::atomic_thread_fence(memory_order_seq_cst);
#  endif
}

_LIBCPP_BEGIN_EXPLICIT_ABI_ANNOTATIONS

[[__gnu__::__returns_nonnull__]] _LIBCPP_AVAILABILITY_HAZARD_POINTER _LIBCPP_EXPORTED_FROM_ABI __hazard_pointer_slot*
__hazard_pointer_acquire();

_LIBCPP_AVAILABILITY_HAZARD_POINTER _LIBCPP_EXPORTED_FROM_ABI void
__hazard_pointer_release([[__gnu__::__nonnull__]] __hazard_pointer_slot* __slot) noexcept;

_LIBCPP_AVAILABILITY_HAZARD_POINTER _LIBCPP_EXPORTED_FROM_ABI void
__hazard_pointer_retire([[__gnu__::__nonnull__]] __hazard_pointer_obj_node* __node) noexcept;

_LIBCPP_END_EXPLICIT_ABI_ANNOTATIONS

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 26 && _LIBCPP_HAS_THREADS

#endif // _LIBCPP___HAZARD_POINTER_DOMAIN_H
