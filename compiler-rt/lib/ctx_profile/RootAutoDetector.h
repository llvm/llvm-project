/*===- RootAutodetector.h- auto-detect roots for ctxprof  -----------------===*\
|*
|* Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
|* See https://llvm.org/LICENSE.txt for license information.
|* SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
|*
\*===----------------------------------------------------------------------===*/

#ifndef CTX_PROFILE_ROOTAUTODETECTOR_H_
#define CTX_PROFILE_ROOTAUTODETECTOR_H_

#include "sanitizer_common/sanitizer_dense_map.h"
#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_stacktrace.h"
#include <pthread.h>
#include <sanitizer/common_interface_defs.h>

using namespace __asan;
using namespace __sanitizer;

namespace __ctx_profile {

/// Capture all the stack traces observed for a specific thread. The "for a
/// specific thread" part is not enforced, but assumed in determineRoots.
class PerThreadCallsiteTrie {
protected:
  /// A trie. A node is the address of a callsite in a function activation. A
  /// child is a callsite in the activation made from the callsite
  /// corresponding to the parent.
  struct Trie final {
    const uptr CallsiteAddress;
    uint64_t Count = 0;
    DenseMap<uptr, Trie> Children;

    Trie(uptr CallsiteAddress = 0) : CallsiteAddress(CallsiteAddress) {}
  };
  Trie TheTrie;

  /// Return the runtime start address of the function that contains the call at
  /// the runtime address CallsiteAddress. May be overriden for easy testing.
  virtual uptr getFctStartAddr(uptr CallsiteAddress) const;

public:
  PerThreadCallsiteTrie(const PerThreadCallsiteTrie &) = delete;
  PerThreadCallsiteTrie(PerThreadCallsiteTrie &&) = default;
  PerThreadCallsiteTrie() = default;

  virtual ~PerThreadCallsiteTrie() = default;

  void insertStack(const StackTrace &ST);

  /// Return the runtime address of root functions, as determined for this
  /// thread, together with the number of samples that included them.
  DenseMap<uptr, uint64_t> determineRoots() const;
};
} // namespace __ctx_profile
#endif
