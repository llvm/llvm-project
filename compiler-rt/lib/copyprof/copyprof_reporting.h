//===-- copyprof_reporting.h ----------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// This file declares the reporting interface that is used to log unnecessary
// copies identified during object destruction.
///
//===----------------------------------------------------------------------===//

#ifndef COPYPROF_REPORTING_H_
#define COPYPROF_REPORTING_H_

#include "sanitizer_common/sanitizer_common.h"

namespace __copyprof {

// Prints a CopyProf report to stderr. `pc` and `bp` are the program counter
// and frame pointer where the copy was destroyed, `obj_size` its flat size in
// bytes, and `did_allocate` whether the copy allocated memory.
void LogCopyProfReport(uptr pc, uptr bp, uptr obj_size, bool did_allocate);

}  // namespace __copyprof

#endif  // COPYPROF_REPORTING_H_
