//===- bolt/Rewrite/RSeqRewriter.cpp --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Basic support for restartable sequences used by tcmalloc. Prevent critical
// section overrides by ignoring optimizations in containing functions.
//
// References:
//   * https://google.github.io/tcmalloc/rseq.html
//   * tcmalloc/internal/percpu_rseq_x86_64.S
//
//===----------------------------------------------------------------------===//

#include "bolt/Core/BinaryFunction.h"
#include "bolt/Rewrite/MetadataRewriter.h"
#include "bolt/Rewrite/MetadataRewriters.h"
#include "llvm/Support/Errc.h"

using namespace llvm;
using namespace bolt;

namespace {

class RSeqRewriter final : public MetadataRewriter {
public:
  RSeqRewriter(StringRef Name, BinaryContext &BC)
      : MetadataRewriter(Name, BC) {}

  Error preCFGInitializer() override {
    for (const BinarySection &Section : BC.allocatableSections()) {
      if (Section.getName() != "__rseq_cs")
        continue;

      auto handleRelocation = [&](const Relocation &Rel, bool IsDynamic) {
        BinaryFunction *BF = nullptr;
        if (Rel.Symbol)
          BF = BC.getFunctionForSymbol(Rel.Symbol);
        else if (Relocation::isRelative(Rel.Type))
          BF = BC.getBinaryFunctionContainingAddress(Rel.Addend);

        if (!BF) {
          BC.errs() << "BOLT-WARNING: no function found matching "
                    << (IsDynamic ? "dynamic " : "")
                    << "relocation in __rseq_cs\n";
        } else if (!BF->isIgnored()) {
          BC.outs() << "BOLT-INFO: restartable sequence reference detected in "
                    << *BF << ". Function will not be optimized\n";
          BF->setIgnored();
        }
      };

      for (const Relocation &Rel : Section.dynamicRelocations())
        handleRelocation(Rel, /*IsDynamic*/ true);

      for (const Relocation &Rel : Section.relocations())
        handleRelocation(Rel, /*IsDynamic*/ false);
    }

    return Error::success();
  }
};

} // namespace

std::unique_ptr<MetadataRewriter>
llvm::bolt::createRSeqRewriter(BinaryContext &BC) {
  return std::make_unique<RSeqRewriter>("rseq-cs-rewriter", BC);
}
