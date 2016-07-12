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

#include <memory>

namespace llvm {
  class StringRef;
}

namespace lld {
namespace elf {
template <class ELFT> class InputSectionBase;
template <class ELFT> class ObjectFile;
template <class ELFT> class SymbolTable;

template <class ELFT> void writeResult(SymbolTable<ELFT> *Symtab);

template <class ELFT> void markLive();

template <class ELFT>
llvm::StringRef getOutputSectionName(InputSectionBase<ELFT> *S);

template <class ELFT>
void reportDiscarded(InputSectionBase<ELFT> *IS,
                     const std::unique_ptr<elf::ObjectFile<ELFT>> &File);
}
}

#endif
