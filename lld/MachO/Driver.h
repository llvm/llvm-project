//===- Driver.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_DRIVER_H
#define LLD_MACHO_DRIVER_H

#include "lld/Common/LLVM.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Option/OptTable.h"
#include "llvm/Support/MemoryBuffer.h"

namespace lld {
namespace macho {

class DylibFile;

class MachOOptTable : public llvm::opt::OptTable {
public:
  MachOOptTable();
  llvm::opt::InputArgList parse(ArrayRef<const char *> argv);
  void printHelp(const char *argv0, bool showHidden) const;
};

// Create enum with OPT_xxx values for each option in Options.td
enum {
  OPT_INVALID = 0,
#define OPTION(_1, _2, ID, _4, _5, _6, _7, _8, _9, _10, _11, _12) OPT_##ID,
#include "Options.inc"
#undef OPTION
};

// Check for both libfoo.dylib and libfoo.tbd (in that order).
llvm::Optional<std::string> resolveDylibPath(llvm::StringRef path);

llvm::Optional<DylibFile *> makeDylibFromTAPI(llvm::MemoryBufferRef mbref,
                                              DylibFile *umbrella = nullptr);

} // namespace macho
} // namespace lld

#endif
