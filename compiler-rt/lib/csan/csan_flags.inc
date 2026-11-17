//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Host ConcurrencySanitizer runtime flags.
///
//===----------------------------------------------------------------------===//

#ifndef CSAN_FLAG
#error "Define CSAN_FLAG prior to including this file!"
#endif

// CSAN_FLAG(Type, Name, DefaultValue, Description)

CSAN_FLAG(int, udelay, 80, "Microseconds to stall after arming a watchpoint.")
CSAN_FLAG(int, skip_watch, 4000,
          "Skip this many accesses (per thread) before arming a watchpoint.")
CSAN_FLAG(bool, halt_on_error, false, "Die after the first reported race.")
