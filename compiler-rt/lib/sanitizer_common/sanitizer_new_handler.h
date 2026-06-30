//===-- sanitizer_new_handler.h ---------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is shared between run-time libraries of sanitizers.
//
// It provides the operator new chain-handling framework required by
// [new.delete.single]/3+/4: a std::get_new_handler() loop plus the
// throwing / nothrow exhaustion policies that compose with it. Each
// sanitizer's operator new wrapper supplies two small lambdas:
//
//   * Alloc        — invokes the sanitizer's internal allocator, returning
//                    nullptr on OOM (never aborting on OOM). Other detected
//                    failure modes (e.g. invalid alignment) should abort with
//                    a diagnostic.
//
//   * OnExhausted  — invokes the sanitizer's "abort with diagnostic"
//                    handler (e.g. asan's ReportOutOfMemory + Die()).
//                    Required to never return (enforced by UNREACHABLE()).
//===----------------------------------------------------------------------===//
#ifndef SANITIZER_NEW_HANDLER_H
#define SANITIZER_NEW_HANDLER_H

#include "sanitizer_allocator.h"
#include "sanitizer_internal_defs.h"
#include "sanitizer_platform.h"

#if !SANITIZER_WINDOWS && defined(__cpp_exceptions)
#  include <new>
#else
// Builds without exception support can't include <new>: on Windows the
// sanitizer runtimes are built without exceptions; elsewhere this TU may
// have been compiled with -fno-exceptions (e.g. with
// -DCOMPILER_RT_ASAN_ENABLE_EXCEPTIONS=OFF). Forward-declare just enough
// of std for the loop; the real std::get_new_handler is supplied by the
// C++ runtime the sanitizer links against (vcruntime / msvcprt /
// MinGW libstdc++ / libstdc++ / libc++) — it doesn't need to come from
// <new>.
namespace std {
struct nothrow_t {};
enum class align_val_t : __sanitizer::usize {};
using new_handler = void (*)();
new_handler get_new_handler() noexcept;
}  // namespace std
#endif  // !SANITIZER_WINDOWS && defined(__cpp_exceptions)

namespace __sanitizer {

// Runs std::get_new_handler() per [new.delete.single]/3+/4 until the
// allocation succeeds or the chain is exhausted. Returns the allocated
// pointer on success, nullptr if the handler chain is exhausted.
//
// NOTE: Exceptions thrown by Alloc or std::new_handler callbacks escape this
//       function. Callers that need to convert exceptions to a nullptr return
//       (e.g. NewImplNothrow below) must wrap the call in try/catch themselves.
template <typename Alloc>
void* RunNewHandlerChain(Alloc alloc) {
  for (;;) {
    void* res = alloc();
    if (LIKELY(res != nullptr))
      return res;
    std::new_handler handler = std::get_new_handler();
    if (!handler)
      return nullptr;
    handler();
  }
}

// NORETURN wrapper providing pre-C++23 compatibility.
template <typename OnExhausted>
NORETURN void InvokeOnExhausted(OnExhausted on_exhausted) {
  on_exhausted();
  UNREACHABLE("operator new OnExhausted callable returned");
}

// Throwing operator new: chain, then on exhaustion throw std::bad_alloc
// when this TU was compiled with exception support and AllocatorMayReturnNull()
// is true. Otherwise (Windows runtimes, -fno-exceptions builds, or the
// default flag value) fall through to the abort path.
template <typename Alloc, typename OnExhausted>
void* NewImplThrowing(Alloc alloc, OnExhausted on_exhausted) {
  void* res = RunNewHandlerChain(alloc);
  if (LIKELY(res != nullptr))
    return res;
#if !SANITIZER_WINDOWS && defined(__cpp_exceptions)
  if (AllocatorMayReturnNull())
    throw std::bad_alloc();
#endif
  InvokeOnExhausted(on_exhausted);
}

// Nothrow operator new: per [new.delete.single]/4 behaves as-if the
// throwing form is called within a try/catch. When exception support is
// absent (Windows runtimes or -fno-exceptions TUs) we can't write the
// try/catch literally — instead we avoid the throw path entirely, so the
// main body matches NewImplThrowing with "throw std::bad_alloc()" replaced
// by "return nullptr".
template <typename Alloc, typename OnExhausted>
void* NewImplNothrow(Alloc alloc, OnExhausted on_exhausted) noexcept {
#if !SANITIZER_WINDOWS && defined(__cpp_exceptions)
  try {
#endif
    void* res = RunNewHandlerChain(alloc);
    if (LIKELY(res != nullptr))
      return res;
    if (AllocatorMayReturnNull())
      return nullptr;
    InvokeOnExhausted(on_exhausted);
#if !SANITIZER_WINDOWS && defined(__cpp_exceptions)
  } catch (...) {
    return nullptr;
  }
#endif
}

}  // namespace __sanitizer

#endif  // SANITIZER_NEW_HANDLER_H
