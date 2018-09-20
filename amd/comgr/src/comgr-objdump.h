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
*******************************************************************************/

#ifndef COMGR_OBJDUMP_H
#define COMGR_OBJDUMP_H

#include "llvm/DebugInfo/DIContext.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/DataTypes.h"
#include "llvm/Object/Archive.h"

// ELFDump.cpp
void printELFFileHeader(const llvm::object::ObjectFile *o);

namespace llvm {
class StringRef;

namespace object {
  class COFFObjectFile;
  class COFFImportFile;
  class MachOObjectFile;
  class ObjectFile;
  class Archive;
  class RelocationRef;
}

extern std::string TripleName;
extern std::string ArchName;
extern std::string MCPU;
extern std::vector<std::string> MAttrs;
extern std::vector<std::string> FilterSections;
extern bool Disassemble;
extern bool DisassembleAll;
extern bool NoShowRawInsn;
extern bool NoLeadingAddr;
extern bool PrivateHeaders;
extern bool FirstPrivateHeader;
extern bool ExportsTrie;
extern bool Rebase;
extern bool Bind;
extern bool LazyBind;
extern bool WeakBind;
extern bool RawClangAST;
extern bool UniversalHeaders;
extern bool ArchiveHeaders;
extern bool IndirectSymbols;
extern bool DataInCode;
extern bool LinkOptHints;
extern bool InfoPlist;
extern bool DylibsUsed;
extern bool DylibId;
extern bool ObjcMetaData;
extern std::string DisSymName;
extern bool NonVerbose;
extern bool Relocations;
extern bool SectionHeaders;
extern bool SectionContents;
extern bool SymbolTable;
extern bool UnwindInfo;
extern bool PrintImmHex;
extern DIDumpType DwarfDumpType;

class DisassemHelper {
  private:
  std::string result_buffer;
  bool byte_stream;

  // Various helper functions.

  // llvm-objdump.cpp
  void DisassembleObject(const object::ObjectFile *Obj, bool InlineRelocs);
  void PrintUnwindInfo(const object::ObjectFile *o);
  void printExportsTrie(const object::ObjectFile *o);
  void printRebaseTable(object::ObjectFile *o);
  void printBindTable(object::ObjectFile *o);
  void printLazyBindTable(object::ObjectFile *o);
  void printWeakBindTable(object::ObjectFile *o);
  void printRawClangAST(const object::ObjectFile *o);
  void PrintRelocations(const object::ObjectFile *o);
  void PrintSectionHeaders(const object::ObjectFile *o);
  void PrintSectionContents(const object::ObjectFile *o);
  void PrintSymbolTable(const object::ObjectFile *o, StringRef ArchiveName,
                        StringRef ArchitectureName = StringRef());
  void printFaultMaps(const object::ObjectFile *Obj);
  void printPrivateFileHeaders(const object::ObjectFile *o, bool onlyFirst);

  void DumpObject(object::ObjectFile *o, const object::Archive *a);
  void DumpArchive(const object::Archive *a);
  void DumpInput(StringRef file);

  // llvm-elfdump.cpp
  //template <class ELFT>
  //void printProgramHeaders(const ELFFile<ELFT> *o);
  void printELFFileHeader(const object::ObjectFile *Obj);

public:
  raw_string_ostream *SOS;

  DisassemHelper(bool byte_dis) : byte_stream(byte_dis) {
    result_buffer.clear();
    SOS = new raw_string_ostream(result_buffer);
  }

  ~DisassemHelper() {
    result_buffer.clear();
    delete SOS;
  }

  std::string &get_result() { return result_buffer; }

  int DisassembleAction(char *inp, size_t in_size, StringRef cpu);
  int DisassembleAction_2(char *inp, size_t in_size);
}; // DisassemHelper

} // end namespace llvm

#endif
