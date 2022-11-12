//===--- LocateSymbol.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AnalysisInternal.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::include_cleaner {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SymbolLocation &S) {
  switch (S.kind()) {
  case SymbolLocation::Physical:
    // We can't decode the Location without SourceManager. Its raw
    // representation isn't completely useless (and distinguishes
    // SymbolReference from Symbol).
    return OS << "@0x"
              << llvm::utohexstr(
                     S.physical().getRawEncoding(), /*LowerCase=*/false,
                     /*Width=*/CHAR_BIT * sizeof(SourceLocation::UIntTy));
  case SymbolLocation::Standard:
    return OS << S.standard().scope() << S.standard().name();
  }
  llvm_unreachable("Unhandled Symbol kind");
}

} // namespace clang::include_cleaner