//===--- TextNodeDumper.cpp - Printing of AST nodes -----------------------===//
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

#include "clang/AST/TextNodeDumper.h"

using namespace clang;

TextNodeDumper::TextNodeDumper(raw_ostream &OS, bool ShowColors,
                               const SourceManager *SM,
                               const PrintingPolicy &PrintPolicy,
                               const comments::CommandTraits *Traits)
    : OS(OS), ShowColors(ShowColors), SM(SM), PrintPolicy(PrintPolicy),
      Traits(Traits) {}

void TextNodeDumper::Visit(const comments::Comment *C,
                           const comments::FullComment *FC) {
  if (!C) {
    ColorScope Color(OS, ShowColors, NullColor);
    OS << "<<<NULL>>>";
    return;
  }

  {
    ColorScope Color(OS, ShowColors, CommentColor);
    OS << C->getCommentKindName();
  }
  dumpPointer(C);
  dumpSourceRange(C->getSourceRange());

  ConstCommentVisitor<TextNodeDumper, void,
                      const comments::FullComment *>::visit(C, FC);
}

void TextNodeDumper::dumpPointer(const void *Ptr) {
  ColorScope Color(OS, ShowColors, AddressColor);
  OS << ' ' << Ptr;
}

void TextNodeDumper::dumpLocation(SourceLocation Loc) {
  if (!SM)
    return;

  ColorScope Color(OS, ShowColors, LocationColor);
  SourceLocation SpellingLoc = SM->getSpellingLoc(Loc);

  // The general format we print out is filename:line:col, but we drop pieces
  // that haven't changed since the last loc printed.
  PresumedLoc PLoc = SM->getPresumedLoc(SpellingLoc);

  if (PLoc.isInvalid()) {
    OS << "<invalid sloc>";
    return;
  }

  if (strcmp(PLoc.getFilename(), LastLocFilename) != 0) {
    OS << PLoc.getFilename() << ':' << PLoc.getLine() << ':'
       << PLoc.getColumn();
    LastLocFilename = PLoc.getFilename();
    LastLocLine = PLoc.getLine();
  } else if (PLoc.getLine() != LastLocLine) {
    OS << "line" << ':' << PLoc.getLine() << ':' << PLoc.getColumn();
    LastLocLine = PLoc.getLine();
  } else {
    OS << "col" << ':' << PLoc.getColumn();
  }
}

void TextNodeDumper::dumpSourceRange(SourceRange R) {
  // Can't translate locations if a SourceManager isn't available.
  if (!SM)
    return;

  OS << " <";
  dumpLocation(R.getBegin());
  if (R.getBegin() != R.getEnd()) {
    OS << ", ";
    dumpLocation(R.getEnd());
  }
  OS << ">";

  // <t2.c:123:421[blah], t2.c:412:321>
}

void TextNodeDumper::dumpBareType(QualType T, bool Desugar) {
  ColorScope Color(OS, ShowColors, TypeColor);

  SplitQualType T_split = T.split();
  OS << "'" << QualType::getAsString(T_split, PrintPolicy) << "'";

  if (Desugar && !T.isNull()) {
    // If the type is sugared, also dump a (shallow) desugared type.
    SplitQualType D_split = T.getSplitDesugaredType();
    if (T_split != D_split)
      OS << ":'" << QualType::getAsString(D_split, PrintPolicy) << "'";
  }
}

void TextNodeDumper::dumpType(QualType T) {
  OS << ' ';
  dumpBareType(T);
}

void TextNodeDumper::dumpBareDeclRef(const Decl *D) {
  if (!D) {
    ColorScope Color(OS, ShowColors, NullColor);
    OS << "<<<NULL>>>";
    return;
  }

  {
    ColorScope Color(OS, ShowColors, DeclKindNameColor);
    OS << D->getDeclKindName();
  }
  dumpPointer(D);

  if (const NamedDecl *ND = dyn_cast<NamedDecl>(D)) {
    ColorScope Color(OS, ShowColors, DeclNameColor);
    OS << " '" << ND->getDeclName() << '\'';
  }

  if (const ValueDecl *VD = dyn_cast<ValueDecl>(D))
    dumpType(VD->getType());
}

void TextNodeDumper::dumpName(const NamedDecl *ND) {
  if (ND->getDeclName()) {
    ColorScope Color(OS, ShowColors, DeclNameColor);
    OS << ' ' << ND->getNameAsString();
  }
}

void TextNodeDumper::dumpAccessSpecifier(AccessSpecifier AS) {
  switch (AS) {
  case AS_none:
    break;
  case AS_public:
    OS << "public";
    break;
  case AS_protected:
    OS << "protected";
    break;
  case AS_private:
    OS << "private";
    break;
  }
}

void TextNodeDumper::dumpCXXTemporary(const CXXTemporary *Temporary) {
  OS << "(CXXTemporary";
  dumpPointer(Temporary);
  OS << ")";
}

const char *TextNodeDumper::getCommandName(unsigned CommandID) {
  if (Traits)
    return Traits->getCommandInfo(CommandID)->Name;
  const comments::CommandInfo *Info =
      comments::CommandTraits::getBuiltinCommandInfo(CommandID);
  if (Info)
    return Info->Name;
  return "<not a builtin command>";
}

void TextNodeDumper::visitTextComment(const comments::TextComment *C,
                                      const comments::FullComment *) {
  OS << " Text=\"" << C->getText() << "\"";
}

void TextNodeDumper::visitInlineCommandComment(
    const comments::InlineCommandComment *C, const comments::FullComment *) {
  OS << " Name=\"" << getCommandName(C->getCommandID()) << "\"";
  switch (C->getRenderKind()) {
  case comments::InlineCommandComment::RenderNormal:
    OS << " RenderNormal";
    break;
  case comments::InlineCommandComment::RenderBold:
    OS << " RenderBold";
    break;
  case comments::InlineCommandComment::RenderMonospaced:
    OS << " RenderMonospaced";
    break;
  case comments::InlineCommandComment::RenderEmphasized:
    OS << " RenderEmphasized";
    break;
  }

  for (unsigned i = 0, e = C->getNumArgs(); i != e; ++i)
    OS << " Arg[" << i << "]=\"" << C->getArgText(i) << "\"";
}

void TextNodeDumper::visitHTMLStartTagComment(
    const comments::HTMLStartTagComment *C, const comments::FullComment *) {
  OS << " Name=\"" << C->getTagName() << "\"";
  if (C->getNumAttrs() != 0) {
    OS << " Attrs: ";
    for (unsigned i = 0, e = C->getNumAttrs(); i != e; ++i) {
      const comments::HTMLStartTagComment::Attribute &Attr = C->getAttr(i);
      OS << " \"" << Attr.Name << "=\"" << Attr.Value << "\"";
    }
  }
  if (C->isSelfClosing())
    OS << " SelfClosing";
}

void TextNodeDumper::visitHTMLEndTagComment(
    const comments::HTMLEndTagComment *C, const comments::FullComment *) {
  OS << " Name=\"" << C->getTagName() << "\"";
}

void TextNodeDumper::visitBlockCommandComment(
    const comments::BlockCommandComment *C, const comments::FullComment *) {
  OS << " Name=\"" << getCommandName(C->getCommandID()) << "\"";
  for (unsigned i = 0, e = C->getNumArgs(); i != e; ++i)
    OS << " Arg[" << i << "]=\"" << C->getArgText(i) << "\"";
}

void TextNodeDumper::visitParamCommandComment(
    const comments::ParamCommandComment *C, const comments::FullComment *FC) {
  OS << " "
     << comments::ParamCommandComment::getDirectionAsString(C->getDirection());

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

  if (C->isParamIndexValid() && !C->isVarArgParam())
    OS << " ParamIndex=" << C->getParamIndex();
}

void TextNodeDumper::visitTParamCommandComment(
    const comments::TParamCommandComment *C, const comments::FullComment *FC) {
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

void TextNodeDumper::visitVerbatimBlockComment(
    const comments::VerbatimBlockComment *C, const comments::FullComment *) {
  OS << " Name=\"" << getCommandName(C->getCommandID())
     << "\""
        " CloseName=\""
     << C->getCloseName() << "\"";
}

void TextNodeDumper::visitVerbatimBlockLineComment(
    const comments::VerbatimBlockLineComment *C,
    const comments::FullComment *) {
  OS << " Text=\"" << C->getText() << "\"";
}

void TextNodeDumper::visitVerbatimLineComment(
    const comments::VerbatimLineComment *C, const comments::FullComment *) {
  OS << " Text=\"" << C->getText() << "\"";
}
