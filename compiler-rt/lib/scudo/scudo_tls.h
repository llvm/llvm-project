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

namespace __scudo {

struct ALIGNED(64) ScudoThreadContext {
 public:
  AllocatorCache Cache;
  Xorshift128Plus Prng;
  uptr QuarantineCachePlaceHolder[4];
  void init();
  void commitBack();
};

void initThread();

// Fastpath functions are defined in the following platform specific headers.
#include "scudo_tls_linux.h"

}  // namespace __scudo

#endif  // SCUDO_TLS_H_
