//===- comgr-objdump.h - Disassemble files to source ----------------------===//
//
// Part of Comgr, under the Apache License v2.0 with LLVM Exceptions. See
// amd/comgr/LICENSE.TXT in this repository for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef COMGR_OBJDUMP_H
#define COMGR_OBJDUMP_H

#include "comgr.h"
#include "llvm/DebugInfo/DIContext.h"
#include "llvm/Object/Archive.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"

namespace llvm {
class StringRef;
}

namespace llvm::object {
class COFFObjectFile;
class COFFImportFile;
class MachOObjectFile;
class ObjectFile;
class Archive;
class RelocationRef;
} // namespace llvm::object

namespace COMGR {
class DisassemHelper {
private:
  llvm::raw_ostream &OutS;
  llvm::raw_ostream &ErrS;

  void DisassembleObject(const llvm::object::ObjectFile *Obj,
                         bool InlineRelocs);
  void printRawClangAST(const llvm::object::ObjectFile *o);
  void PrintRelocations(const llvm::object::ObjectFile *o);
  void PrintSectionHeaders(const llvm::object::ObjectFile *o);
  void PrintSectionContents(const llvm::object::ObjectFile *o);
  void PrintSymbolTable(const llvm::object::ObjectFile *o,
                        llvm::StringRef ArchiveName,
                        llvm::StringRef ArchitectureName = llvm::StringRef());
  void printFaultMaps(const llvm::object::ObjectFile *Obj);
  void printPrivateFileHeaders(const llvm::object::ObjectFile *o,
                               bool onlyFirst);

  void DumpObject(llvm::object::ObjectFile *o, const llvm::object::Archive *a);
  void DumpArchive(const llvm::object::Archive *a);
  void DumpInput(llvm::StringRef file);

  void printELFFileHeader(const llvm::object::ObjectFile *Obj);

public:
  DisassemHelper(llvm::raw_ostream &OutS, llvm::raw_ostream &ErrS)
      : OutS(OutS), ErrS(ErrS) {}

  amd_comgr_status_t disassembleAction(llvm::StringRef Input,
                                       llvm::ArrayRef<std::string> Options);
}; // DisassemHelper

} // end namespace COMGR

#endif
