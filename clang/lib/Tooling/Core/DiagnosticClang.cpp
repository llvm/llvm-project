//===--- Diagnostic.cpp - Framework for clang diagnostics tools ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  Implements classes to support/store diagnostics refactoring. This file
//  contains all of the code that depends on Clang, so that Diagnostic.cpp can
//  be used from tools used to build Clang, like tblgen.
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/SourceLocation.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Tooling/Core/Diagnostic.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
namespace tooling {

DiagnosticMessage::DiagnosticMessage(llvm::StringRef Message,
                                     const SourceManager &Sources,
                                     SourceLocation Loc)
    : Message(Message), FileOffset(0) {
  assert(Loc.isValid() && Loc.isFileID());
  FilePath = std::string(Sources.getFilename(Loc));

  // Don't store offset in the scratch space. It doesn't tell anything to the
  // user. Moreover, it depends on the history of macro expansions and thus
  // prevents deduplication of warnings in headers.
  if (!FilePath.empty())
    FileOffset = Sources.getFileOffset(Loc);
}

FileByteRange::FileByteRange(const SourceManager &Sources,
                             CharSourceRange Range)
    : FileOffset(0), Length(0) {
  FilePath = std::string(Sources.getFilename(Range.getBegin()));
  if (!FilePath.empty()) {
    FileOffset = Sources.getFileOffset(Range.getBegin());
    Length = Sources.getFileOffset(Range.getEnd()) - FileOffset;
  }
}

} // end namespace tooling
} // end namespace clang
