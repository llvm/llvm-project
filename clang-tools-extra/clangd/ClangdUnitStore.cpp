//===--- ClangdUnitStore.cpp - A ClangdUnits container -----------*-C++-*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ClangdUnitStore.h"
#include "llvm/Support/Path.h"
#include <algorithm>

using namespace clang::clangd;
using namespace clang;

std::shared_ptr<CppFile> CppFileCollection::removeIfPresent(PathRef File) {
  std::lock_guard<std::mutex> Lock(Mutex);

  auto It = OpenedFiles.find(File);
  if (It == OpenedFiles.end())
    return nullptr;

  std::shared_ptr<CppFile> Result = It->second;
  OpenedFiles.erase(It);
  return Result;
}
