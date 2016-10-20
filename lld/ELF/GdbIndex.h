//===- GdbIndex.h --------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===-------------------------------------------------------------------===//

#ifndef LLD_ELF_GDB_INDEX_H
#define LLD_ELF_GDB_INDEX_H

#include "InputFiles.h"
#include "llvm/Object/ELF.h"

namespace lld {
namespace elf {

template <class ELFT> class InputSection;

template <class ELFT>
std::vector<std::pair<typename ELFT::uint, typename ELFT::uint>>
readCuList(InputSection<ELFT> *Sec);

} // namespace elf
} // namespace lld

#endif
