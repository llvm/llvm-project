//===-- scudo_tls_linux.inc -------------------------------------*- C++ -*-===//
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

#if SANITIZER_LINUX && !SANITIZER_ANDROID

enum ThreadState : u8 {
  ThreadNotInitialized = 0,
  ThreadInitialized,
  ThreadTornDown,
};
__attribute__((tls_model("initial-exec")))
extern THREADLOCAL ThreadState ScudoThreadState;
__attribute__((tls_model("initial-exec")))
extern THREADLOCAL ScudoThreadContext ThreadLocalContext;

ALWAYS_INLINE void initThreadMaybe() {
  if (LIKELY(ScudoThreadState != ThreadNotInitialized))
    return;
  initThread();
}

ALWAYS_INLINE ScudoThreadContext *getThreadContextAndLock() {
  if (UNLIKELY(ScudoThreadState == ThreadTornDown))
    return nullptr;
  return &ThreadLocalContext;
}

#endif  // SANITIZER_LINUX && !SANITIZER_ANDROID

#endif  // SCUDO_TLS_LINUX_H_
