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

namespace lld {
namespace elf {

template <class ELFT> class SymbolTable;

template <class ELFT> void writeResult(SymbolTable<ELFT> *Symtab);

template <class ELFT> void markLive();
}
}

#endif
