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
                               const PrintingPolicy &PrintPolicy)
    : OS(OS), ShowColors(ShowColors), SM(SM), PrintPolicy(PrintPolicy) {}

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
