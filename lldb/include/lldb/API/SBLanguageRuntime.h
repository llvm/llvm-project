//===-- SBLanguageRuntime.h -------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_API_SBLANGUAGERUNTIME_H
#define LLDB_API_SBLANGUAGERUNTIME_H

#include "lldb/API/SBDefines.h"

namespace lldb {

class SBLanguageRuntime {
public:
  static lldb::LanguageType GetLanguageTypeFromString(const char *string);

  static const char *GetNameForLanguageType(lldb::LanguageType language);
};

} // namespace lldb

#endif // LLDB_API_SBLANGUAGERUNTIME_H
