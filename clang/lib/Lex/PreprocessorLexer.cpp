//===- PreprocessorLexer.cpp - C Language Family Lexer --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements the PreprocessorLexer and Token interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/PreprocessorLexer.h"
#include "clang/Basic/SourceManager.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/Token.h"
#include <cassert>

using namespace clang;

void PreprocessorLexer::anchor() {}

PreprocessorLexer::PreprocessorLexer(Preprocessor *pp, FileID fid)
    : PP(pp), FID(fid) {
  if (pp)
    InitialNumSLocEntries = pp->getSourceManager().local_sloc_entry_size();
}

/// After the preprocessor has parsed a \#include, lex and
/// (potentially) macro expand the filename.
void PreprocessorLexer::LexIncludeFilename(Token &FilenameTok) {
  assert(ParsingPreprocessorDirective &&
         ParsingFilename == false &&
         "Must be in a preprocessing directive!");

  // We are now parsing a filename!
  ParsingFilename = true;

  // Lex the filename.
  if (LexingRawMode)
    IndirectLex(FilenameTok);
  else
    PP->Lex(FilenameTok);

  // We should have obtained the filename now.
  ParsingFilename = false;

  // No filename?
  if (FilenameTok.is(tok::eod))
    PP->Diag(FilenameTok.getLocation(), diag::err_pp_expects_filename);
}

/// getFileEntry - Return the FileEntry corresponding to this FileID.  Like
/// getFileID(), this only works for lexers with attached preprocessors.
const FileEntry *PreprocessorLexer::getFileEntry() const {
  return PP->getSourceManager().getFileEntryForID(getFileID());
}
