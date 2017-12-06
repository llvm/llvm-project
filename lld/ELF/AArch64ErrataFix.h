//===- AArch64ErrataFix.h ---------------------------------------*- C++ -*-===//
//
//                             The LLVM Linker
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_AARCH64ERRATAFIX_H
#define LLD_ELF_AARCH64ERRATAFIX_H

#include "lld/Common/LLVM.h"

namespace lld {
namespace elf {

class OutputSection;
void reportA53Errata843419Fixes();

} // namespace elf
} // namespace lld

#endif
