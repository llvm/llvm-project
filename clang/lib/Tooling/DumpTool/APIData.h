//===- APIData.h ---------------------------------------------*- C++ -*----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_TOOLING_DUMPTOOL_APIDATA_H
#define LLVM_CLANG_LIB_TOOLING_DUMPTOOL_APIDATA_H

#include <string>
#include <vector>

namespace clang {
namespace tooling {

struct ClassData {

  bool isEmpty() const {
    return ASTClassLocations.empty() && ASTClassRanges.empty();
  }

  std::vector<std::string> ASTClassLocations;
  std::vector<std::string> ASTClassRanges;
  // TODO: Extend this with locations available via typelocs etc.
};

} // namespace tooling
} // namespace clang

#endif
