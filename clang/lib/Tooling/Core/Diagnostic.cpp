//===--- Diagnostic.cpp - Framework for clang diagnostics tools ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implements classes to support/store diagnostics refactoring.
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/Core/Diagnostic.h"
#include "clang/Basic/SourceManager.h"

namespace clang {
namespace tooling {

DiagnosticMessage::DiagnosticMessage(llvm::StringRef Message)
    : Message(Message), FileOffset(0) {}

DiagnosticMessage::DiagnosticMessage(llvm::StringRef Message,
                                     const SourceManager &Sources,
                                     SourceLocation Loc)
    : Message(Message), FileOffset(0) {
  assert(Loc.isValid() && Loc.isFileID());
  FilePath = Sources.getFilename(Loc);

  // Don't store offset in the scratch space. It doesn't tell anything to the
  // user. Moreover, it depends on the history of macro expansions and thus
  // prevents deduplication of warnings in headers.
  if (!FilePath.empty())
    FileOffset = Sources.getFileOffset(Loc);
}

Diagnostic::Diagnostic(llvm::StringRef DiagnosticName,
                       Diagnostic::Level DiagLevel, StringRef BuildDirectory)
    : DiagnosticName(DiagnosticName), DiagLevel(DiagLevel),
      BuildDirectory(BuildDirectory) {}

Diagnostic::Diagnostic(llvm::StringRef DiagnosticName,
                       const DiagnosticMessage &Message,
                       const llvm::StringMap<Replacements> &Fix,
                       const SmallVector<DiagnosticMessage, 1> &Notes,
                       Level DiagLevel, llvm::StringRef BuildDirectory)
    : DiagnosticName(DiagnosticName), Message(Message), Fix(Fix), Notes(Notes),
      DiagLevel(DiagLevel), BuildDirectory(BuildDirectory) {}

} // end namespace tooling
} // end namespace clang
