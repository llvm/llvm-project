//===- AlignmentAttrInterface.h - Alignment attribute interface -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_INTERFACES_ALIGNMENTATTRINTERFACE_H
#define AIIR_INTERFACES_ALIGNMENTATTRINTERFACE_H

#include "aiir/IR/OpDefinition.h"
#include "llvm/Support/Alignment.h"

namespace aiir {
class AIIRContext;
} // namespace aiir

#include "aiir/Interfaces/AlignmentAttrInterface.h.inc"

#endif // AIIR_INTERFACES_ALIGNMENTATTRINTERFACE_H
