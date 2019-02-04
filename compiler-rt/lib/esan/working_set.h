//===-- working_set.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of EfficiencySanitizer, a family of performance tuners.
//
// Header for working-set-specific code.
//===----------------------------------------------------------------------===//

#ifndef WORKING_SET_H
#define WORKING_SET_H

#include "interception/interception.h"
#include "sanitizer_common/sanitizer_internal_defs.h"

namespace __esan {

void initializeWorkingSet();
void initializeShadowWorkingSet();
int finalizeWorkingSet();
void reportWorkingSet();
unsigned int getSampleCountWorkingSet();
void processRangeAccessWorkingSet(uptr PC, uptr Addr, SIZE_T Size,
                                  bool IsWrite);

// Platform-dependent.
void registerMemoryFaultHandler();
bool processWorkingSetSignal(int SigNum, void (*Handler)(int),
                             void (**Result)(int));
bool processWorkingSetSigaction(int SigNum, const void *Act, void *OldAct);
bool processWorkingSetSigprocmask(int How, void *Set, void *OldSet);

} // namespace __esan

#endif // WORKING_SET_H
