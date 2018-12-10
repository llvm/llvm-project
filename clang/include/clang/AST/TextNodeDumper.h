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
#include "clang/AST/CommentCommandTraits.h"
#include "clang/AST/CommentVisitor.h"
#include "clang/AST/ExprCXX.h"

namespace clang {

class TextNodeDumper
    : public comments::ConstCommentVisitor<TextNodeDumper, void,
                                           const comments::FullComment *> {
  raw_ostream &OS;
  const bool ShowColors;

  /// Keep track of the last location we print out so that we can
  /// print out deltas from then on out.
  const char *LastLocFilename = "";
  unsigned LastLocLine = ~0U;

  const SourceManager *SM;

  /// The policy to use for printing; can be defaulted.
  PrintingPolicy PrintPolicy;

  const comments::CommandTraits *Traits;

  const char *getCommandName(unsigned CommandID);

public:
  TextNodeDumper(raw_ostream &OS, bool ShowColors, const SourceManager *SM,
                 const PrintingPolicy &PrintPolicy,
                 const comments::CommandTraits *Traits);

  void Visit(const comments::Comment *C, const comments::FullComment *FC);

  void dumpPointer(const void *Ptr);
  void dumpLocation(SourceLocation Loc);
  void dumpSourceRange(SourceRange R);
  void dumpBareType(QualType T, bool Desugar = true);
  void dumpType(QualType T);
  void dumpBareDeclRef(const Decl *D);
  void dumpName(const NamedDecl *ND);
  void dumpAccessSpecifier(AccessSpecifier AS);
  void dumpCXXTemporary(const CXXTemporary *Temporary);

  void visitTextComment(const comments::TextComment *C,
                        const comments::FullComment *);
  void visitInlineCommandComment(const comments::InlineCommandComment *C,
                                 const comments::FullComment *);
  void visitHTMLStartTagComment(const comments::HTMLStartTagComment *C,
                                const comments::FullComment *);
  void visitHTMLEndTagComment(const comments::HTMLEndTagComment *C,
                              const comments::FullComment *);
  void visitBlockCommandComment(const comments::BlockCommandComment *C,
                                const comments::FullComment *);
  void visitParamCommandComment(const comments::ParamCommandComment *C,
                                const comments::FullComment *FC);
  void visitTParamCommandComment(const comments::TParamCommandComment *C,
                                 const comments::FullComment *FC);
  void visitVerbatimBlockComment(const comments::VerbatimBlockComment *C,
                                 const comments::FullComment *);
  void
  visitVerbatimBlockLineComment(const comments::VerbatimBlockLineComment *C,
                                const comments::FullComment *);
  void visitVerbatimLineComment(const comments::VerbatimLineComment *C,
                                const comments::FullComment *);
};

} // namespace clang

#endif // LLVM_CLANG_AST_TEXTNODEDUMPER_H
