//=-- dsan_fuchsia.h ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This file is a part of DoubleFreeSanitizer.
// Standalone DSan RTL code specific to Fuchsia.
//
//===---------------------------------------------------------------------===//

#ifndef DSAN_FUCHSIA_H
#define DSAN_FUCHSIA_H

#include "dsan_thread.h"
#include "sanitizer_common/sanitizer_platform.h"

#if !SANITIZER_FUCHSIA
#  error "dsan_fuchsia.h is used only on Fuchsia systems (SANITIZER_FUCHSIA)"
#endif

namespace __dsan {

class ThreadContext final : public ThreadContextDsanBase {
 public:
  explicit ThreadContext(int tid);
  void OnCreated(void* arg) override;
  void OnStarted(void* arg) override;
};

}  // namespace __dsan

#endif  // DSAN_FUCHSIA_H
