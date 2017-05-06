//===-- scudo_tls_context_linux.inc -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Linux specific base thread context definition.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_TLS_CONTEXT_LINUX_INC_
#define SCUDO_TLS_CONTEXT_LINUX_INC_

#ifndef SCUDO_TLS_H_
# error "This file must be included inside scudo_tls.h."
#endif  // SCUDO_TLS_H_

#if SANITIZER_LINUX && !SANITIZER_ANDROID

struct ScudoThreadContextPlatform {
  ALWAYS_INLINE void unlock() {}
};

#endif  // SANITIZER_LINUX && !SANITIZER_ANDROID

#endif  // SCUDO_TLS_CONTEXT_LINUX_INC_
