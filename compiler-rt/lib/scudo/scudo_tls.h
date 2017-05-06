//===-- scudo_tls.h ---------------------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// Scudo thread local structure definition.
/// Implementation will differ based on the thread local storage primitives
/// offered by the underlying platform.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_TLS_H_
#define SCUDO_TLS_H_

#include "scudo_allocator.h"
#include "scudo_utils.h"

#include "sanitizer_common/sanitizer_linux.h"
#include "sanitizer_common/sanitizer_platform.h"

namespace __scudo {

// Platform specific base thread context definitions.
#include "scudo_tls_context_android.inc"
#include "scudo_tls_context_linux.inc"

struct ALIGNED(64) ScudoThreadContext : public ScudoThreadContextPlatform {
  AllocatorCache Cache;
  Xorshift128Plus Prng;
  uptr QuarantineCachePlaceHolder[4];
  void init();
  void commitBack();
};

void initThread();

// Platform specific dastpath functions definitions.
#include "scudo_tls_android.inc"
#include "scudo_tls_linux.inc"

}  // namespace __scudo

#endif  // SCUDO_TLS_H_
