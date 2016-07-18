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
namespace elf {

class InputFile;
struct Symbol;

enum ELFKind {
  ELFNoneKind,
  ELF32LEKind,
  ELF32BEKind,
  ELF64LEKind,
  ELF64BEKind
};

enum class BuildIdKind { None, Fnv1, Md5, Sha1, Hexstring };

enum class UnresolvedPolicy { NoUndef, Error, Warn, Ignore };

struct SymbolVersion {
  llvm::StringRef Name;
  bool IsExternCpp;
};

// This struct contains symbols version definition that
// can be found in version script if it is used for link.
struct VersionDefinition {
  VersionDefinition(llvm::StringRef Name, size_t Id) : Name(Name), Id(Id) {}
  llvm::StringRef Name;
  size_t Id;
  std::vector<SymbolVersion> Globals;
  size_t NameOff; // Offset in string table.
};

// This struct contains the global configuration for the linker.
// Most fields are direct mapping from the command line options
// and such fields have the same name as the corresponding options.
// Most fields are initialized by the driver.
struct Configuration {
  Symbol *EntrySym = nullptr;
  InputFile *FirstElf = nullptr;
  llvm::StringRef DynamicLinker;
  llvm::StringRef Entry;
  llvm::StringRef Emulation;
  llvm::StringRef Fini;
  llvm::StringRef Init;
  llvm::StringRef LtoAAPipeline;
  llvm::StringRef LtoNewPmPasses;
  llvm::StringRef OutputFile;
  llvm::StringRef SoName;
  llvm::StringRef Sysroot;
  std::string RPath;
  std::vector<VersionDefinition> VersionDefinitions;
  std::vector<llvm::StringRef> DynamicList;
  std::vector<llvm::StringRef> SearchPaths;
  std::vector<llvm::StringRef> Undefined;
  std::vector<SymbolVersion> VersionScriptGlobals;
  std::vector<uint8_t> BuildIdVector;
  bool AllowMultipleDefinition;
  bool AsNeeded = false;
  bool Bsymbolic;
  bool BsymbolicFunctions;
  bool Demangle = true;
  bool DisableVerify;
  bool DiscardAll;
  bool DiscardLocals;
  bool DiscardNone;
  bool EhFrameHdr;
  bool EnableNewDtags;
  bool ExportDynamic;
  bool FatalWarnings;
  bool GcSections;
  bool GnuHash = false;
  bool ICF;
  bool Mips64EL = false;
  bool NoGnuUnique;
  bool NoUndefinedVersion;
  bool Pic;
  bool Pie;
  bool PrintGcSections;
  bool Rela;
  bool Relocatable;
  bool SaveTemps;
  bool Shared;
  bool Static = false;
  bool StripAll;
  bool StripDebug;
  bool SysvHash = true;
  bool Threads;
  bool Trace;
  bool Verbose;
  bool WarnCommon;
  bool ZCombreloc;
  bool ZExecStack;
  bool ZNodelete;
  bool ZNow;
  bool ZOrigin;
  bool ZRelro;
  UnresolvedPolicy UnresolvedSymbols;
  BuildIdKind BuildId = BuildIdKind::None;
  ELFKind EKind = ELFNoneKind;
  uint16_t DefaultSymbolVersion = llvm::ELF::VER_NDX_GLOBAL;
  uint16_t EMachine = llvm::ELF::EM_NONE;
  uint64_t EntryAddr = -1;
  uint64_t ImageBase;
  unsigned LtoJobs;
  unsigned LtoO;
  unsigned Optimize;
};

// The only instance of Configuration struct.
extern Configuration *Config;

} // namespace elf
} // namespace lld

#endif
