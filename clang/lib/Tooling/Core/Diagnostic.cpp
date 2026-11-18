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
#include "llvm/ADT/STLExtras.h"

using llvm::SmallVector;
using llvm::StringRef;

namespace clang {
namespace tooling {

DiagnosticMessage::DiagnosticMessage(llvm::StringRef Message)
    : Message(Message), FileOffset(0) {}

Diagnostic::Diagnostic(llvm::StringRef DiagnosticName,
                       Diagnostic::Level DiagLevel, StringRef BuildDirectory)
    : DiagnosticName(DiagnosticName), DiagLevel(DiagLevel),
      BuildDirectory(BuildDirectory) {}

Diagnostic::Diagnostic(llvm::StringRef DiagnosticName,
                       const DiagnosticMessage &Message,
                       const SmallVector<DiagnosticMessage, 1> &Notes,
                       Level DiagLevel, llvm::StringRef BuildDirectory)
    : DiagnosticName(DiagnosticName), Message(Message), Notes(Notes),
      DiagLevel(DiagLevel), BuildDirectory(BuildDirectory) {}

const llvm::StringMap<Replacements> *selectFirstFix(const Diagnostic& D) {
   if (!D.Message.Fix.empty())
    return &D.Message.Fix;
  auto Iter = llvm::find_if(D.Notes, [](const tooling::DiagnosticMessage &D) {
    return !D.Fix.empty();
  });
  if (Iter != D.Notes.end())
    return &Iter->Fix;
  return nullptr;
}

} // end namespace tooling
} // end namespace clang
