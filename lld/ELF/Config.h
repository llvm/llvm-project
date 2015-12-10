//===- Config.h -------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_CONFIG_H
#define LLD_ELF_CONFIG_H

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/ELF.h"

#include <vector>

namespace lld {
namespace elf2 {

class InputFile;
class SymbolBody;

enum ELFKind {
  ELFNoneKind,
  ELF32LEKind,
  ELF32BEKind,
  ELF64LEKind,
  ELF64BEKind
};

struct Configuration {
  SymbolBody *EntrySym = nullptr;
  InputFile *FirstElf = nullptr;
  llvm::StringRef DynamicLinker;
  llvm::StringRef Entry;
  llvm::StringRef Emulation;
  llvm::StringRef Fini;
  llvm::StringRef Init;
  llvm::StringRef OutputFile;
  llvm::StringRef SoName;
  llvm::StringRef Sysroot;
  std::string RPath;
  llvm::MapVector<llvm::StringRef, std::vector<llvm::StringRef>> OutputSections;
  std::vector<llvm::StringRef> SearchPaths;
  std::vector<llvm::StringRef> Undefined;
  bool AllowMultipleDefinition;
  bool AsNeeded = false;
  bool Bsymbolic;
  bool DiscardAll;
  bool DiscardLocals;
  bool DiscardNone;
  bool EnableNewDtags;
  bool ExportDynamic;
  bool GcSections;
  bool GnuHash = false;
  bool Mips64EL = false;
  bool NoInhibitExec;
  bool NoUndefined;
  bool PrintGcSections;
  bool Shared;
  bool Static = false;
  bool StripAll;
  bool SysvHash = true;
  bool Verbose;
  bool ZExecStack;
  bool ZNodelete;
  bool ZNow;
  bool ZOrigin;
  bool ZRelro;
  ELFKind EKind = ELFNoneKind;
  uint16_t EMachine = llvm::ELF::EM_NONE;
  uint64_t EntryAddr = -1;
  unsigned Optimize = 0;
};

extern Configuration *Config;

} // namespace elf2
} // namespace lld

#endif
