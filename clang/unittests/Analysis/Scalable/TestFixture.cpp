//===- unittests/Analysis/Scalable/TestFixture.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "TestFixture.h"
#include "clang/Analysis/Scalable/Model/EntityId.h"
#include "llvm/Support/raw_ostream.h"
#include <ostream>
#include <string>

using namespace clang;
using namespace ssaf;

template <class T> static std::string asString(const T &Obj) {
  std::string Repr;
  llvm::raw_string_ostream(Repr) << Obj;
  return Repr;
}

void TestFixture::PrintTo(const EntityId &E, std::ostream *OS) {
  *OS << "EntityId(" << E.Index << ")";
}
void TestFixture::PrintTo(const SummaryName &N, std::ostream *OS) {
  *OS << "SummaryName(" << N.Name << ")";
}
