//===- EhFrame.h ------------------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_EHFRAME_H
#define LLD_ELF_EHFRAME_H

#include "lld/Core/LLVM.h"

namespace lld {
namespace elf {
template <class ELFT> size_t readEhRecordSize(ArrayRef<uint8_t> Data);
template <class ELFT> uint8_t getFdeEncoding(ArrayRef<uint8_t> Data);
}
}

#endif
