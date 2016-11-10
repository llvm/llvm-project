//===- Writer.h -------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_WRITER_H
#define LLD_ELF_WRITER_H

#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <memory>

namespace lld {
namespace elf {
class InputFile;
class OutputSectionBase;
template <class ELFT> class InputSectionBase;
template <class ELFT> class ObjectFile;
template <class ELFT> class SymbolTable;
template <class ELFT> void writeResult();
template <class ELFT> void markLive();
template <class ELFT> bool isRelroSection(const OutputSectionBase *Sec);

// This describes a program header entry.
// Each contains type, access flags and range of output sections that will be
// placed in it.
template <class ELFT> struct PhdrEntry {
  PhdrEntry(unsigned Type, unsigned Flags);
  void add(OutputSectionBase *Sec);

  typename ELFT::Phdr H = {};
  OutputSectionBase *First = nullptr;
  OutputSectionBase *Last = nullptr;
  bool HasLMA = false;
};

llvm::StringRef getOutputSectionName(llvm::StringRef Name);

template <class ELFT> void reportDiscarded(InputSectionBase<ELFT> *IS);

template <class ELFT> uint32_t getMipsEFlags();

uint8_t getMipsFpAbiFlag(uint8_t OldFlag, uint8_t NewFlag,
                         llvm::StringRef FileName);

bool isMipsN32Abi(const InputFile *F);
}
}

#endif
