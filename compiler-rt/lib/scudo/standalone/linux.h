//===-- linux.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SCUDO_LINUX_H_
#define SCUDO_LINUX_H_

#include "platform.h"

#if SCUDO_LINUX

namespace scudo {

// MapPlatformData is unused on Linux, define it as a minimally sized structure.
struct MapPlatformData {};

#if SCUDO_ANDROID

#if defined(__aarch64__)
#define __get_tls()                                                            \
  ({                                                                           \
    void **__v;                                                                \
    __asm__("mrs %0, tpidr_el0" : "=r"(__v));                                  \
    __v;                                                                       \
  })
#elif defined(__arm__)
#define __get_tls()                                                            \
  ({                                                                           \
    void **__v;                                                                \
    __asm__("mrc p15, 0, %0, c13, c0, 3" : "=r"(__v));                         \
    __v;                                                                       \
  })
#elif defined(__i386__)
#define __get_tls()                                                            \
  ({                                                                           \
    void **__v;                                                                \
    __asm__("movl %%gs:0, %0" : "=r"(__v));                                    \
    __v;                                                                       \
  })
#elif defined(__x86_64__)
#define __get_tls()                                                            \
  ({                                                                           \
    void **__v;                                                                \
    __asm__("mov %%fs:0, %0" : "=r"(__v));                                     \
    __v;                                                                       \
  })
#else
#error "Unsupported architecture."
#endif

// The Android Bionic team has allocated a TLS slot for sanitizers starting
// with Q, given that Android currently doesn't support ELF TLS. It is used to
// store sanitizer thread specific data.
static const int TLS_SLOT_SANITIZER = 6;

ALWAYS_INLINE uptr *getAndroidTlsPtr() {
  return reinterpret_cast<uptr *>(&__get_tls()[TLS_SLOT_SANITIZER]);
}

#endif // SCUDO_ANDROID

} // namespace scudo

#endif // SCUDO_LINUX

#endif // SCUDO_LINUX_H_
