//===- ICF.h --------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_ICF_H
#define LLD_ELF_ICF_H

#include "Target.h"
namespace lld::elf {
struct Ctx;
class TargetInfo;

template <class ELFT> void doIcf(Ctx &);
}

#endif
