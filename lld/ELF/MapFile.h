//===- MapFile.h ------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_MAPFILE_H
#define LLD_ELF_MAPFILE_H

#include "OutputSections.h"

namespace lld {
namespace elf {
template <class ELFT>
void writeMapFile(llvm::ArrayRef<OutputSectionBase *> OutputSections);
}
}

#endif
