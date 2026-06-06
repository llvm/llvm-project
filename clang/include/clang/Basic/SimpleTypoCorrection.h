//===- SimpleTypoCorrection.h - Basic typo correction utility -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SimpleTypoCorrection class, which performs basic
// typo correction using string similarity based on edit distance.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_BASIC_SIMPLETYPOCORRECTION_H
#define LLVM_CLANG_BASIC_SIMPLETYPOCORRECTION_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"

namespace clang {

class IdentifierInfo;

class SimpleTypoCorrection {
  StringRef BestCandidate;
  StringRef Typo;

  const unsigned MaxEditDistance;
  unsigned BestEditDistance;
  unsigned BestIndex;
  unsigned NextIndex;

public:
  explicit SimpleTypoCorrection(StringRef Typo)
      : BestCandidate(), Typo(Typo), MaxEditDistance((Typo.size() + 2) / 3),
        BestEditDistance(MaxEditDistance + 1), BestIndex(0), NextIndex(0) {}

  void add(const StringRef Candidate);
  void add(const char *Candidate);
  void add(const IdentifierInfo *Candidate);

  std::optional<StringRef> getCorrection() const;
  bool hasCorrection() const;
  unsigned getCorrectionIndex() const;
};
} // namespace clang

#endif // LLVM_CLANG_BASIC_SIMPLETYPOCORRECTION_H
