/*******************************************************************************
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright (c) 2003-2017 University of Illinois at Urbana-Champaign.
 * Modifications (c) 2018 Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * with the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 *     * Redistributions of source code must retain the above copyright notice,
 *       this list of conditions and the following disclaimers.
 *
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimers in the
 *       documentation and/or other materials provided with the distribution.
 *
 *     * Neither the names of the LLVM Team, University of Illinois at
 *       Urbana-Champaign, nor the names of its contributors may be used to
 *       endorse or promote products derived from this Software without specific
 *       prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS WITH
 * THE SOFTWARE.
 *
 ******************************************************************************/

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
} // end namespace object

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
