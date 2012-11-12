//===--- CommentDumper.cpp - Dumping implementation for Comment ASTs ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/CommentVisitor.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace comments {

namespace {
class CommentDumper: public comments::ConstCommentVisitor<CommentDumper> {
  raw_ostream &OS;
  const CommandTraits *Traits;
  const SourceManager *SM;

  /// The \c FullComment parent of the comment being dumped.
  const FullComment *FC;

  unsigned IndentLevel;

public:
  CommentDumper(raw_ostream &OS,
                const CommandTraits *Traits,
                const SourceManager *SM,
                const FullComment *FC) :
      OS(OS), Traits(Traits), SM(SM), FC(FC), IndentLevel(0)
  { }

  void dumpIndent() const {
    for (unsigned i = 1, e = IndentLevel; i < e; ++i)
      OS << "  ";
  }

  void dumpLocation(SourceLocation Loc) {
    if (SM)
      Loc.print(OS, *SM);
  }

  void dumpSourceRange(const Comment *C);

  void dumpComment(const Comment *C);

  void dumpSubtree(const Comment *C);

  // Inline content.
  void visitTextComment(const TextComment *C);
  void visitInlineCommandComment(const InlineCommandComment *C);
  void visitHTMLStartTagComment(const HTMLStartTagComment *C);
  void visitHTMLEndTagComment(const HTMLEndTagComment *C);

  // Block content.
  void visitParagraphComment(const ParagraphComment *C);
  void visitBlockCommandComment(const BlockCommandComment *C);
  void visitParamCommandComment(const ParamCommandComment *C);
  void visitTParamCommandComment(const TParamCommandComment *C);
  void visitVerbatimBlockComment(const VerbatimBlockComment *C);
  void visitVerbatimBlockLineComment(const VerbatimBlockLineComment *C);
  void visitVerbatimLineComment(const VerbatimLineComment *C);

  void visitFullComment(const FullComment *C);

  const char *getCommandName(unsigned CommandID) {
    if (Traits)
      return Traits->getCommandInfo(CommandID)->Name;
    const CommandInfo *Info = CommandTraits::getBuiltinCommandInfo(CommandID);
    if (Info)
      return Info->Name;
    return "<not a builtin command>";
  }
};

void CommentDumper::dumpSourceRange(const Comment *C) {
  if (!SM)
    return;

  SourceRange SR = C->getSourceRange();

  OS << " <";
  dumpLocation(SR.getBegin());
  if (SR.getBegin() != SR.getEnd()) {
    OS << ", ";
    dumpLocation(SR.getEnd());
  }
  OS << ">";
}

void CommentDumper::dumpComment(const Comment *C) {
  dumpIndent();
  OS << "(" << C->getCommentKindName()
     << " " << (const void *) C;
  dumpSourceRange(C);
}

void CommentDumper::dumpSubtree(const Comment *C) {
  ++IndentLevel;
  if (C) {
    visit(C);
    for (Comment::child_iterator I = C->child_begin(),
                                 E = C->child_end();
         I != E; ++I) {
      OS << '\n';
      dumpSubtree(*I);
    }
    OS << ')';
  } else {
    dumpIndent();
    OS << "<<<NULL>>>";
  }
  --IndentLevel;
}

void CommentDumper::visitTextComment(const TextComment *C) {
  dumpComment(C);

  OS << " Text=\"" << C->getText() << "\"";
}

void CommentDumper::visitInlineCommandComment(const InlineCommandComment *C) {
  dumpComment(C);

  OS << " Name=\"" << getCommandName(C->getCommandID()) << "\"";
  switch (C->getRenderKind()) {
  case InlineCommandComment::RenderNormal:
    OS << " RenderNormal";
    break;
  case InlineCommandComment::RenderBold:
    OS << " RenderBold";
    break;
  case InlineCommandComment::RenderMonospaced:
    OS << " RenderMonospaced";
    break;
  case InlineCommandComment::RenderEmphasized:
    OS << " RenderEmphasized";
    break;
  }

  for (unsigned i = 0, e = C->getNumArgs(); i != e; ++i)
    OS << " Arg[" << i << "]=\"" << C->getArgText(i) << "\"";
}

void CommentDumper::visitHTMLStartTagComment(const HTMLStartTagComment *C) {
  dumpComment(C);

  OS << " Name=\"" << C->getTagName() << "\"";
  if (C->getNumAttrs() != 0) {
    OS << " Attrs: ";
    for (unsigned i = 0, e = C->getNumAttrs(); i != e; ++i) {
      const HTMLStartTagComment::Attribute &Attr = C->getAttr(i);
      OS << " \"" << Attr.Name << "=\"" << Attr.Value << "\"";
    }
  }
  if (C->isSelfClosing())
    OS << " SelfClosing";
}

void CommentDumper::visitHTMLEndTagComment(const HTMLEndTagComment *C) {
  dumpComment(C);

  OS << " Name=\"" << C->getTagName() << "\"";
}

void CommentDumper::visitParagraphComment(const ParagraphComment *C) {
  dumpComment(C);
}

void CommentDumper::visitBlockCommandComment(const BlockCommandComment *C) {
  dumpComment(C);

  OS << " Name=\"" << getCommandName(C->getCommandID()) << "\"";
  for (unsigned i = 0, e = C->getNumArgs(); i != e; ++i)
    OS << " Arg[" << i << "]=\"" << C->getArgText(i) << "\"";
}

void CommentDumper::visitParamCommandComment(const ParamCommandComment *C) {
  dumpComment(C);

  OS << " " << ParamCommandComment::getDirectionAsString(C->getDirection());

  if (C->isDirectionExplicit())
    OS << " explicitly";
  else
    OS << " implicitly";

  if (C->hasParamName()) {
    if (C->isParamIndexValid())
      OS << " Param=\"" << C->getParamName(FC) << "\"";
    else
      OS << " Param=\"" << C->getParamNameAsWritten() << "\"";
  }

  if (C->isParamIndexValid())
    OS << " ParamIndex=" << C->getParamIndex();
}

void CommentDumper::visitTParamCommandComment(const TParamCommandComment *C) {
  dumpComment(C);

  if (C->hasParamName()) {
    if (C->isPositionValid())
      OS << " Param=\"" << C->getParamName(FC) << "\"";
    else
      OS << " Param=\"" << C->getParamNameAsWritten() << "\"";
  }

  if (C->isPositionValid()) {
    OS << " Position=<";
    for (unsigned i = 0, e = C->getDepth(); i != e; ++i) {
      OS << C->getIndex(i);
      if (i != e - 1)
        OS << ", ";
    }
    OS << ">";
  }
}

void CommentDumper::visitVerbatimBlockComment(const VerbatimBlockComment *C) {
  dumpComment(C);

  OS << " Name=\"" << getCommandName(C->getCommandID()) << "\""
        " CloseName=\"" << C->getCloseName() << "\"";
}

void CommentDumper::visitVerbatimBlockLineComment(const VerbatimBlockLineComment *C) {
  dumpComment(C);

  OS << " Text=\"" << C->getText() << "\"";
}

void CommentDumper::visitVerbatimLineComment(const VerbatimLineComment *C) {
  dumpComment(C);

  OS << " Text=\"" << C->getText() << "\"";
}

void CommentDumper::visitFullComment(const FullComment *C) {
  dumpComment(C);
}

} // unnamed namespace

void Comment::dump(llvm::raw_ostream &OS, const CommandTraits *Traits,
                   const SourceManager *SM) const {
  const FullComment *FC = dyn_cast<FullComment>(this);
  CommentDumper D(llvm::errs(), Traits, SM, FC);
  D.dumpSubtree(this);
  llvm::errs() << '\n';
}

} // end namespace comments
} // end namespace clang

