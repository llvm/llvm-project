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
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::include_cleaner {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Symbol &S) {
  switch (S.kind()) {
  case Symbol::Declaration:
    if (const auto *ND = llvm::dyn_cast<NamedDecl>(&S.declaration()))
      return OS << ND->getNameAsString();
    return OS << S.declaration().getDeclKindName();
  case Symbol::Macro:
    return OS << S.macro().Name;
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
  case Header::Verbatim:
    return OS << H.verbatim();
  }
  llvm_unreachable("Unhandled Header kind");
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const Include &I) {
  return OS << I.Line << ": " << I.Spelled << " => "
            << (I.Resolved ? I.Resolved->getName() : "<missing>");
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SymbolReference &R) {
  // We can't decode the Location without SourceManager. Its raw representation
  // isn't completely useless (and distinguishes SymbolReference from Symbol).
  return OS << R.Target << "@0x"
            << llvm::utohexstr(
                   R.RefLocation.getRawEncoding(), /*LowerCase=*/false,
                   /*Width=*/CHAR_BIT * sizeof(SourceLocation::UIntTy));
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, RefType T) {
  switch (T) {
  case RefType::Explicit:
    return OS << "explicit";
  case RefType::Implicit:
    return OS << "implicit";
  case RefType::Ambiguous:
    return OS << "ambiguous";
  }
  llvm_unreachable("Unexpected RefType");
}

} // namespace clang::include_cleaner
