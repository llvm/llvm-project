//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_HOST_WINDOWS_PATHUTILS_H
#define LLDB_HOST_WINDOWS_PATHUTILS_H

#include "llvm/ADT/StringRef.h"

#include <string>

namespace lldb_private {

/// Convert an extended-length path to a regular DOS path: "\\?\C:\a" becomes
/// "C:\a" and "\\?\UNC\server\share" becomes "\\server\share". Other paths are
/// returned unchanged.
inline std::string StripExtendedLengthPrefix(llvm::StringRef path) {
  if (path.consume_front("\\\\?\\UNC\\"))
    return "\\\\" + path.str();
  path.consume_front("\\\\?\\");
  return path.str();
}

} // namespace lldb_private

#endif // LLDB_HOST_WINDOWS_PATHUTILS_H
