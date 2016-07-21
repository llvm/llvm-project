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

#include <cstdint>
#include <memory>

namespace llvm {
  class StringRef;
}

namespace lld {
namespace elf {
template <class ELFT> class OutputSectionBase;
template <class ELFT> class InputSectionBase;
template <class ELFT> class ObjectFile;
template <class ELFT> class SymbolTable;
template <class ELFT> void writeResult(SymbolTable<ELFT> *Symtab);
template <class ELFT> void markLive();
template <class ELFT> bool isOutputDynamic();
template <class ELFT> bool isRelroSection(OutputSectionBase<ELFT> *Sec);
template <class ELFT> bool needsPtLoad(OutputSectionBase<ELFT> *Sec);
uint32_t toPhdrFlags(uint64_t Flags);

// This describes a program header entry.
// Each contains type, access flags and range of output sections that will be
// placed in it.
template<class ELFT>
struct PhdrEntry {
  PhdrEntry(unsigned Type, unsigned Flags);
  void add(OutputSectionBase<ELFT> *Sec);

  typename ELFT::Phdr H = {};
  OutputSectionBase<ELFT> *First = nullptr;
  OutputSectionBase<ELFT> *Last = nullptr;
};

template <class ELFT>
llvm::StringRef getOutputSectionName(InputSectionBase<ELFT> *S);

template <class ELFT>
void reportDiscarded(InputSectionBase<ELFT> *IS,
                     const std::unique_ptr<elf::ObjectFile<ELFT>> &File);
}
}

#endif
