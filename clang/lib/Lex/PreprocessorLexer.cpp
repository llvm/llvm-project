//===--- PreprocessorLexer.cpp - C Language Family Lexer ------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements the PreprocessorLexer and Token interfaces.
//
//===----------------------------------------------------------------------===//

#include "clang/Lex/PreprocessorLexer.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Lex/LexDiagnostic.h"
#include "clang/Basic/SourceManager.h"
using namespace clang;

void PreprocessorLexer::anchor() { }

PreprocessorLexer::PreprocessorLexer(Preprocessor *pp, FileID fid)
  : PP(pp), FID(fid), InitialNumSLocEntries(0),
    ParsingPreprocessorDirective(false),
    ParsingFilename(false), LexingRawMode(false) {
  if (pp)
    InitialNumSLocEntries = pp->getSourceManager().local_sloc_entry_size();
}

/// \brief After the preprocessor has parsed a \#include, lex and
/// (potentially) macro expand the filename.
void PreprocessorLexer::LexIncludeFilename(Token &FilenameTok) {
  assert(ParsingPreprocessorDirective &&
         ParsingFilename == false &&
         "Must be in a preprocessing directive!");

  // We are now parsing a filename!
  ParsingFilename = true;

  // Lex the filename.
  IndirectLex(FilenameTok);

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
