//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_FORMAT_H
#define LLDB_UTILITY_FORMAT_H

#include "lldb/lldb-enumerations.h"

namespace lldb_private {

bool GetFormatFromCString(const char *name, lldb::Format &format);
char GetFormatAsFormatChar(lldb::Format format);
const char *GetFormatAsCString(lldb::Format format);

} // namespace lldb_private

#endif // LLDB_UTILITY_FORMAT_H
