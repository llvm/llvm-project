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

#include <optional>

namespace lld::elf {
struct EhSectionPiece;

uint8_t getFdeEncoding(EhSectionPiece *p);
// Returns the 'P' (personality) encoding of a CIE, if it has one.
std::optional<uint8_t> getPersonalityEncoding(const EhSectionPiece &p,
                                              bool reportErrors = true);
// reportErrors=false suppresses diagnostics (used by pre-passes that run
// before EhFrameSection::finalizeContents, which reports them).
bool hasLSDA(const EhSectionPiece &p, bool reportErrors = true);
}

#endif
