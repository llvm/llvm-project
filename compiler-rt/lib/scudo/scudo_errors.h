//===-- scudo_errors.h ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Header for scudo_errors.cpp.
///
//===----------------------------------------------------------------------===//

#ifndef SCUDO_ERRORS_H_
#define SCUDO_ERRORS_H_

#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __scudo {

void NORETURN reportCallocOverflow(uptr Count, uptr Size);
void NORETURN reportPvallocOverflow(uptr Size);
void NORETURN reportAllocationAlignmentTooBig(uptr Alignment,
                                              uptr MaxAlignment);
void NORETURN reportAllocationAlignmentNotPowerOfTwo(uptr Alignment);
void NORETURN reportInvalidPosixMemalignAlignment(uptr Alignment);
void NORETURN reportInvalidAlignedAllocAlignment(uptr Size, uptr Alignment);
void NORETURN reportAllocationSizeTooBig(uptr UserSize, uptr TotalSize,
                                         uptr MaxSize);
void NORETURN reportRssLimitExceeded();
void NORETURN reportOutOfMemory(uptr RequestedSize);

}  // namespace __scudo

#endif  // SCUDO_ERRORS_H_
