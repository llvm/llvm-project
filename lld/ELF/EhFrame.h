//===- EhFrame.h ------------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_ELF_EHFRAME_H
#define LLD_ELF_EHFRAME_H

#include "lld/Common/LLVM.h"

namespace lld::elf {
class EhInputSection;
struct EhSectionPiece;

uint8_t getFdeEncoding(const EhInputSection &sec, const EhSectionPiece &p);
bool hasLSDA(const EhInputSection &sec, const EhSectionPiece &p);
bool hasZeroPcRange(const EhInputSection &sec, const EhSectionPiece &p,
                    uint8_t enc);
}

#endif
