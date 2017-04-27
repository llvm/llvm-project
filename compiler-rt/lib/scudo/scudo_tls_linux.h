//===-- scudo_tls_linux.h ---------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Scudo thread local structure fastpath functions implementation for platforms
/// supporting thread_local.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_TLS_LINUX_H_
#define SCUDO_TLS_LINUX_H_

#ifndef SCUDO_TLS_H_
# error "This file must be included inside scudo_tls.h."
#endif  // SCUDO_TLS_H_

#include "sanitizer_common/sanitizer_platform.h"

#if SANITIZER_LINUX

enum ThreadState : u8 {
  ThreadNotInitialized = 0,
  ThreadInitialized,
  ThreadTornDown,
};
extern thread_local ThreadState ScudoThreadState;
extern thread_local ScudoThreadContext ThreadLocalContext;

ALWAYS_INLINE void initThreadMaybe() {
  if (LIKELY(ScudoThreadState != ThreadNotInitialized))
    return;
  initThread();
}

ALWAYS_INLINE ScudoThreadContext *getThreadContext() {
  if (UNLIKELY(ScudoThreadState == ThreadTornDown))
    return nullptr;
  return &ThreadLocalContext;
}

#endif  // SANITIZER_LINUX

#endif  // SCUDO_TLS_LINUX_H_
