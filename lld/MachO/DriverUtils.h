//===- DriverUtils.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_DRIVER_UTILS_H
#define LLD_MACHO_DRIVER_UTILS_H

#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

namespace lld {
namespace macho {

class DylibFile;

// Check for both libfoo.dylib and libfoo.tbd (in that order).
llvm::Optional<std::string> resolveDylibPath(llvm::StringRef path);

llvm::Optional<DylibFile *> makeDylibFromTAPI(llvm::MemoryBufferRef mbref,
                                              DylibFile *umbrella = nullptr);

} // namespace macho
} // namespace lld

#endif
