//===-- tsan_symbolize.h ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file is a part of ThreadSanitizer (TSan), a race detector.
//
//===----------------------------------------------------------------------===//
#ifndef TSAN_SYMBOLIZE_H
#define TSAN_SYMBOLIZE_H

#include "tsan_defs.h"
#include "tsan_report.h"

namespace __tsan {

SymbolizedStack *SymbolizeCode(uptr addr);
bool SymbolizeData(uptr addr, ReportLocation* loc);
void SymbolizerFlush();

}  // namespace __tsan

#endif  // TSAN_SYMBOLIZE_H
