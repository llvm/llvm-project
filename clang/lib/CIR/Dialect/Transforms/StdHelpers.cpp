//===- StdHelpers.cpp - Implementation standard related helpers--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "StdHelpers.h"

namespace mlir {
namespace cir {

bool isStdArrayType(mlir::Type t) {
  auto sTy = t.dyn_cast<StructType>();
  if (!sTy)
    return false;
  auto recordDecl = sTy.getAst();
  if (!recordDecl.isInStdNamespace())
    return false;

  // TODO: only std::array supported for now, generalize and
  // use tablegen. CallDescription.cpp in the static analyzer
  // could be a good inspiration source too.
  if (recordDecl.getName().compare("array") != 0)
    return false;

  return true;
}

} // namespace cir
} // namespace mlir