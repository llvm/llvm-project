//===--- SemaAPINotesInternal.h - API Notes Sema Internals ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_SEMA_SEMAAPINOTESINTERNAL_H
#define LLVM_CLANG_LIB_SEMA_SEMAAPINOTESINTERNAL_H

#include "clang/APINotes/Types.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include <string>

namespace clang {
namespace api_notes {
class APINotesReader;
}

struct APINotesSelectorDiagnosticEntry {
  api_notes::APINotesFunctionSelectorKey BroadKey;
  llvm::SmallVector<std::string, 4> Parameters;
  bool Used = false;
};

struct APINotesSelectorDiagnosticName {
  SourceLocation Loc;
  std::string Name;
};

struct APINotesSelectorDiagnosticReaderState {
  bool Initialized = false;
  llvm::DenseMap<api_notes::APINotesFunctionSelectorKey, unsigned>
      SelectorIndices;
  llvm::DenseMap<api_notes::APINotesFunctionSelectorKey,
                 APINotesSelectorDiagnosticName>
      SeenNames;
  llvm::SmallVector<APINotesSelectorDiagnosticEntry, 4> Selectors;
};

struct APINotesSelectorDiagnosticState {
  llvm::DenseMap<api_notes::APINotesReader *,
                 APINotesSelectorDiagnosticReaderState>
      Readers;
};

} // namespace clang

#endif // LLVM_CLANG_LIB_SEMA_SEMAAPINOTESINTERNAL_H
