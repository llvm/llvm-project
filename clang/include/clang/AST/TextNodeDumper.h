//===--- TextNodeDumper.h - Printing of AST nodes -------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements AST dumping of components of individual AST nodes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_AST_TEXTNODEDUMPER_H
#define LLVM_CLANG_AST_TEXTNODEDUMPER_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/ASTDumperUtils.h"
#include "clang/AST/ExprCXX.h"

namespace clang {

class TextNodeDumper {
  raw_ostream &OS;
  const bool ShowColors;

  /// Keep track of the last location we print out so that we can
  /// print out deltas from then on out.
  const char *LastLocFilename = "";
  unsigned LastLocLine = ~0U;

  const SourceManager *SM;

  /// The policy to use for printing; can be defaulted.
  PrintingPolicy PrintPolicy;

public:
  TextNodeDumper(raw_ostream &OS, bool ShowColors, const SourceManager *SM,
                 const PrintingPolicy &PrintPolicy);

  void dumpPointer(const void *Ptr);
  void dumpLocation(SourceLocation Loc);
  void dumpSourceRange(SourceRange R);
  void dumpBareType(QualType T, bool Desugar = true);
  void dumpType(QualType T);
  void dumpBareDeclRef(const Decl *D);
  void dumpName(const NamedDecl *ND);
  void dumpAccessSpecifier(AccessSpecifier AS);
  void dumpCXXTemporary(const CXXTemporary *Temporary);
};

} // namespace clang

#endif // LLVM_CLANG_AST_TEXTNODEDUMPER_H
