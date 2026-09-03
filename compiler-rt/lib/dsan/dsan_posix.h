//=-- dsan_posix.h -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file is a part of DoubleFreeSanitizer.
// Standalone DSan RTL code common to POSIX-like systems.
//
//===---------------------------------------------------------------------===//

#ifndef DSAN_POSIX_H
#define DSAN_POSIX_H

#include "dsan_thread.h"
#include "sanitizer_common/sanitizer_platform.h"

#if !SANITIZER_POSIX
#  error "dsan_posix.h is used only on POSIX-like systems (SANITIZER_POSIX)"
#endif

namespace __sanitizer {
struct DTLS;
}

namespace __dsan {

class ThreadContext final : public ThreadContextDsanBase {
 public:
  explicit ThreadContext(int tid);
  void OnStarted(void* arg) override;
  uptr tls_begin() { return tls_begin_; }
  uptr tls_end() { return tls_end_; }
  DTLS* dtls() { return dtls_; }

 private:
  uptr tls_begin_ = 0;
  uptr tls_end_ = 0;
  DTLS* dtls_ = nullptr;
};

void ThreadStart(u32 tid, ThreadID os_id,
                 ThreadType thread_type = ThreadType::Regular);

}  // namespace __dsan

#endif  // DSAN_POSIX_H
