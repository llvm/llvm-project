//===- ViewOpGraph.h - View/write op graphviz graphs ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines interface to produce Graphviz outputs of AIIR op within block.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TRANSFORMS_VIEWOPGRAPH_H_
#define AIIR_TRANSFORMS_VIEWOPGRAPH_H_

#include "aiir/Support/LLVM.h"
#include "llvm/Support/raw_ostream.h"

namespace aiir {
class Pass;

#define GEN_PASS_DECL_VIEWOPGRAPHPASS
#include "aiir/Transforms/Passes.h.inc"

/// Creates a pass to print op graphs with the specified output stream.
std::unique_ptr<Pass> createViewOpGraphPass(raw_ostream &os);

} // namespace aiir

#endif // AIIR_TRANSFORMS_VIEWOPGRAPH_H_
