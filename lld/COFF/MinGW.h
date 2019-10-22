//===- MinGW.h --------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_COFF_MINGW_H
#define LLD_COFF_MINGW_H

#include "Config.h"
#include "Symbols.h"
#include "lld/Common/LLVM.h"

namespace lld {
namespace coff {

// Logic for deciding what symbols to export, when exporting all
// symbols for MinGW.
class AutoExporter {
public:
  AutoExporter();

  void addWholeArchive(StringRef Path);

  llvm::StringSet<> ExcludeSymbols;
  llvm::StringSet<> ExcludeSymbolPrefixes;
  llvm::StringSet<> ExcludeSymbolSuffixes;
  llvm::StringSet<> ExcludeLibs;
  llvm::StringSet<> ExcludeObjects;

  bool shouldExport(Defined *Sym) const;
};

void writeDefFile(StringRef Name);

} // namespace coff
} // namespace lld

#endif
