//===- LinkerOptimizationHints.h ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLD_MACHO_LINKER_OPTIMIZATION_HINTS_H
#define LLD_MACHO_LINKER_OPTIMIZATION_HINTS_H

#include "InputFiles.h"

namespace lld::macho {
void applyOptimizationHints(uint8_t *outBuf, const ObjFile &obj);
}
#endif
