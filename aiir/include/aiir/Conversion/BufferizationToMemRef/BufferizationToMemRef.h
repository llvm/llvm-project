//===- BufferizationToMemRef.h - Bufferization to MemRef conversion -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_BUFFERIZATIONTOMEMREF_BUFFERIZATIONTOMEMREF_H
#define AIIR_CONVERSION_BUFFERIZATIONTOMEMREF_BUFFERIZATIONTOMEMREF_H

#include "aiir/Pass/Pass.h"
#include <memory>

namespace aiir {
class ModuleOp;

#define GEN_PASS_DECL_CONVERTBUFFERIZATIONTOMEMREFPASS
#include "aiir/Conversion/Passes.h.inc"

} // namespace aiir

#endif // AIIR_CONVERSION_BUFFERIZATIONTOMEMREF_BUFFERIZATIONTOMEMREF_H
