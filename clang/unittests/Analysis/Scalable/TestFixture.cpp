//===- unittests/Analysis/Scalable/TestFixture.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFixture.h"
#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "clang/Analysis/Scalable/Model/EntityName.h"
#include "clang/Analysis/Scalable/Model/SummaryName.h"
#include <ostream>
#include <string>

namespace clang::ssaf {

template <class T> static std::string asString(const T &Obj) {
  std::string S;
  llvm::raw_string_ostream(S) << Obj;
  return S;
}

void PrintTo(const BuildNamespace &BN, std::ostream *OS) {
  *OS << asString(BN);
}

void PrintTo(const EntityId &EI, std::ostream *OS) { *OS << asString(EI); }

void PrintTo(const EntityLinkage &EL, std::ostream *OS) { *OS << asString(EL); }

void PrintTo(const EntityName &EN, std::ostream *OS) { *OS << asString(EN); }

void PrintTo(const NestedBuildNamespace &NBN, std::ostream *OS) {
  *OS << asString(NBN);
}

void PrintTo(const SummaryName &SN, std::ostream *OS) { *OS << asString(SN); }

} // namespace clang::ssaf
