//===- PointerFlow.cpp ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Analyses/PointerFlow/PointerFlow.h"

using namespace clang;
using namespace ssaf;

ssaf::PointerFlowEntitySummary
ssaf::buildPointerFlowEntitySummary(EdgeSet Edges) {
  return PointerFlowEntitySummary(std::move(Edges));
}

llvm::iterator_range<EdgeSet::const_iterator>
ssaf::getEdges(const PointerFlowEntitySummary &Sum) {
  return Sum.Edges;
}
