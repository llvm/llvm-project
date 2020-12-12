//===- APIData.h ---------------------------------------------*- C++ -*----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
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
