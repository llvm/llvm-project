//===- Utils.cpp ------------------------------------------------*- C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// The file defines utils functions that can be shared across archs.
//
//===----------------------------------------------------------------------===//

#include "lld/Common/Utils.h"

using namespace llvm;
using namespace lld;

StringRef lld::utils::getRootSymbol(StringRef name) {
  name.consume_back(".Tgm");
  auto [P0, S0] = name.rsplit(".llvm.");
  auto [P1, S1] = P0.rsplit(".__uniq.");
  return P1;
}
