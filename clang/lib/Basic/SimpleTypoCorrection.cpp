//===- SimpleTypoCorrection.cpp - Basic typo correction utility -----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the SimpleTypoCorrection class, which performs basic
// typo correction using string similarity based on edit distance.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SimpleTypoCorrection.h"
#include "clang/Basic/IdentifierTable.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/StringRef.h"

using namespace clang;

void SimpleTypoCorrection::add(const StringRef Candidate) {
  if (Candidate.empty())
    return;

  unsigned MinPossibleEditDistance =
      abs(static_cast<int>(Candidate.size()) - static_cast<int>(Typo.size()));

  if (MinPossibleEditDistance > 0 && Typo.size() / MinPossibleEditDistance < 3)
    return;

  unsigned EditDistance = Typo.edit_distance(
      Candidate, /*AllowReplacements*/ true, MaxEditDistance);

  if (EditDistance < BestEditDistance) {
    BestCandidate = Candidate;
    BestEditDistance = EditDistance;
    BestIndex = NextIndex;
  }

  ++NextIndex;
}

void SimpleTypoCorrection::add(const char *Candidate) {
  if (Candidate)
    add(StringRef(Candidate));
}

void SimpleTypoCorrection::add(const IdentifierInfo *Candidate) {
  if (Candidate)
    add(Candidate->getName());
}

unsigned SimpleTypoCorrection::getCorrectionIndex() const { return BestIndex; }

std::optional<StringRef> SimpleTypoCorrection::getCorrection() const {
  if (hasCorrection())
    return BestCandidate;
  return std::nullopt;
}

bool SimpleTypoCorrection::hasCorrection() const {
  return BestEditDistance <= MaxEditDistance;
}
