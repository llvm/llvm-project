//===-- DumpRegisterInfo.h --------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_CORE_DUMPREGISTERINFO_H
#define LLDB_CORE_DUMPREGISTERINFO_H

#include "lldb/Register/DumpRegisterInfo.h"
#include <stdint.h>
#include <utility>
#include <vector>

namespace lldb_private {

class Stream;
class ThreadRegisterContext;
struct RegisterInfo;
class RegisterType;

void DumpRegisterInfo(Stream &strm, ThreadRegisterContext &ctx,
                      const RegisterInfo &info, uint32_t terminal_width);

} // namespace lldb_private

#endif // LLDB_CORE_DUMPREGISTERINFO_H
