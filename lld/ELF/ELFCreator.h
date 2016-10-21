//===- ELFCreator.h ---------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_ELF_CREATOR_H
#define LLD_ELF_ELF_CREATOR_H

#include "lld/Core/LLVM.h"
#include <string>
#include <vector>

namespace lld {
namespace elf {

// Wraps a given binary blob with an ELF header so that the blob
// can be linked as an ELF file. Used for "--format binary".
template <class ELFT>
std::vector<uint8_t> wrapBinaryWithElfHeader(llvm::ArrayRef<uint8_t> Data,
                                             std::string Filename);
}
}

#endif
