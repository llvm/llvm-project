//===-- UriParser.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLDB_UTILITY_URIPARSER_H
#define LLDB_UTILITY_URIPARSER_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"

namespace lldb_private {
class UriParser {
public:
  // Parses
  // RETURN VALUE
  //   if url is valid, function returns true and
  //   scheme/hostname/port/path are set to the parsed values
  //   port it set to llvm::None if it is not included in the URL
  //
  //   if the url is invalid, function returns false and
  //   output parameters remain unchanged
  static bool Parse(llvm::StringRef uri, llvm::StringRef &scheme,
                    llvm::StringRef &hostname, llvm::Optional<uint16_t> &port,
                    llvm::StringRef &path);
};
}

#endif // LLDB_UTILITY_URIPARSER_H
