//===- HeaderFile.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/InstallAPI/HeaderFile.h"

using namespace llvm;
namespace clang::installapi {

llvm::Regex HeaderFile::getFrameworkIncludeRule() {
  return llvm::Regex("/(.+)\\.framework/(.+)?Headers/(.+)");
}

std::optional<std::string> createIncludeHeaderName(const StringRef FullPath) {
  // Headers in usr(/local)*/include.
  std::string Pattern = "/include/";
  auto PathPrefix = FullPath.find(Pattern);
  if (PathPrefix != StringRef::npos) {
    PathPrefix += Pattern.size();
    return FullPath.drop_front(PathPrefix).str();
  }

  // Framework Headers.
  SmallVector<StringRef, 4> Matches;
  HeaderFile::getFrameworkIncludeRule().match(FullPath, &Matches);
  // Returned matches are always in stable order.
  if (Matches.size() != 4)
    return std::nullopt;

  return Matches[1].drop_front(Matches[1].rfind('/') + 1).str() + "/" +
         Matches[3].str();
}
} // namespace clang::installapi
