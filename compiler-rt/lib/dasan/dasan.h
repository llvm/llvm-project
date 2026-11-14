//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Private interface of the device address sanitizer's host runtime.
///
//===----------------------------------------------------------------------===//

#ifndef DASAN_H
#define DASAN_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_mutex.h"

namespace __dasan {

using namespace __sanitizer;

void Initialize();

// Platform, allocator, and images. Taken inside the RPC server's lock to print
// a report, so never hold it across Offload::TrackExecutable.
extern Mutex DasanMutex;

}  // namespace __dasan

extern "C" {

SANITIZER_INTERFACE_ATTRIBUTE void __dasan_init();

SANITIZER_INTERFACE_ATTRIBUTE const char* __dasan_default_options();

}  // extern "C"

#endif  // DASAN_H
