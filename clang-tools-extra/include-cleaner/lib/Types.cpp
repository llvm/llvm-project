//===--- Types.cpp --------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang-include-cleaner/Types.h"
#include "clang/AST/Decl.h"
#include "clang/Basic/FileEntry.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::include_cleaner {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Symbol &S) {
  switch (S.kind()) {
  case Symbol::Declaration:
    if (const auto *ND = llvm::dyn_cast<NamedDecl>(&S.declaration()))
      return OS << ND->getNameAsString();
    return OS << S.declaration().getDeclKindName();
  case Symbol::Standard:
    return OS << S.standard().scope() << S.standard().name();
  }
  llvm_unreachable("Unhandled Symbol kind");
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Header &H) {
  switch (H.kind()) {
  case Header::Physical:
    return OS << H.physical()->getName();
  case Header::Standard:
    return OS << H.standard().name();
  }
  llvm_unreachable("Unhandled Header kind");
}

} // namespace clang::include_cleaner
