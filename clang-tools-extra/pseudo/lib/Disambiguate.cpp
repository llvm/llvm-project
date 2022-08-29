//===--- Disambiguate.cpp - Find the best tree in the forest --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-pseudo/Disambiguate.h"

namespace clang::pseudo {

Disambiguation disambiguate(const ForestNode *Root,
                            const DisambiguateParams &Params) {
  // FIXME: this is a dummy placeholder strategy, implement a real one!
  Disambiguation Result;
  for (const ForestNode &N : Root->descendants()) {
    if (N.kind() == ForestNode::Ambiguous)
      Result.try_emplace(&N, 1);
  }
  return Result;
}

void removeAmbiguities(ForestNode *&Root, const Disambiguation &D) {
  std::vector<ForestNode **> Queue = {&Root};
  while (!Queue.empty()) {
    ForestNode **Next = Queue.back();
    Queue.pop_back();
    switch ((*Next)->kind()) {
    case ForestNode::Sequence:
      for (ForestNode *&Child : (*Next)->elements())
        Queue.push_back(&Child);
      break;
    case ForestNode::Ambiguous: {
      assert(D.count(*Next) != 0 && "disambiguation is incomplete!");
      ForestNode *ChosenChild = (*Next)->alternatives()[D.lookup(*Next)];
      *Next = ChosenChild;
      Queue.push_back(Next);
      break;
    }
    case ForestNode::Terminal:
    case ForestNode::Opaque:
      break;
    }
  }
}

} // namespace clang::pseudo
