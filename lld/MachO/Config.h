//===- Config.h -------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_CONFIG_H
#define LLD_MACHO_CONFIG_H

#include "llvm/ADT/StringRef.h"
#include "llvm/BinaryFormat/MachO.h"

#include <vector>

namespace lld {
namespace macho {

class Symbol;

struct Configuration {
  Symbol *entry;
  bool hasReexports = false;
  llvm::StringRef installName;
  llvm::StringRef outputFile;
  llvm::MachO::HeaderFileType outputType;
  std::vector<llvm::StringRef> searchPaths;
};

extern Configuration *config;

} // namespace macho
} // namespace lld

#endif
