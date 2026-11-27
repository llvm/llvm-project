//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Tracking associated with executable images. Support for global values comes
/// from loading the executable within the virtual address region and marking
/// chunks that overlap with globals in the executable.
///
//===----------------------------------------------------------------------===//

#ifndef DASAN_IMAGE_H
#define DASAN_IMAGE_H

#include "sanitizer_common/sanitizer_internal_defs.h"
#include "sanitizer_common/sanitizer_offload_hsa.h"

namespace __dasan {

using namespace __sanitizer;

struct GlobalInfo {
  uptr Beg;
  uptr Size;
  const char* Name;
};

// Executable mapping large enough for LoadSize. Addr and Span receive the
// range the loader places into. Empty on failure. The handle stays here.
bool ReserveExecutable(hsa_agent_t Agent, uptr LoadSize, uptr& Addr,
                       uptr& Span);
// Drop a reservation the loader did not keep.
void ReleaseExecutable(uptr Addr);
// The load landed. Freeze may publish its globals.
void RecordExecutable(uptr Addr);

void TrackExecutable(hsa_executable_t Exec);
void UntrackExecutable(hsa_executable_t Exec);
void UntrackAll();

bool LookupGlobal(uptr Beg, GlobalInfo& Out);

}  // namespace __dasan

#endif  // DASAN_IMAGE_H
