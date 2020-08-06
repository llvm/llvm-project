//===--- FileIndexRecord.cpp - Index data per file --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "FileIndexRecord.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/Path.h"

using namespace clang;
using namespace clang::index;

void FileIndexRecord::addDeclOccurence(SymbolRoleSet Roles, unsigned Offset,
                                       const Decl *D,
                                       ArrayRef<SymbolRelation> Relations) {
  assert(D->isCanonicalDecl() &&
         "Occurrences should be associated with their canonical decl");

  auto IsNextOccurence = [&]() -> bool {
    if (Decls.empty())
      return true;
    auto &Last = Decls.back();
    return Last.Offset < Offset;
  };

  if (IsNextOccurence()) {
    Decls.emplace_back(Roles, Offset, D, Relations);
    return;
  }

  DeclOccurrence NewInfo(Roles, Offset, D, Relations);
  // We keep Decls in order as we need to access them in this order in all cases.
  auto It = llvm::upper_bound(Decls, NewInfo);
  Decls.insert(It, std::move(NewInfo));
}

void FileIndexRecord::print(llvm::raw_ostream &OS) const {
  OS << "DECLS BEGIN ---\n";
  for (auto &DclInfo : Decls) {
    const Decl *D = DclInfo.Dcl;
    SourceManager &SM = D->getASTContext().getSourceManager();
    SourceLocation Loc = SM.getFileLoc(D->getLocation());
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    OS << llvm::sys::path::filename(PLoc.getFilename()) << ':' << PLoc.getLine()
       << ':' << PLoc.getColumn();

    if (auto ND = dyn_cast<NamedDecl>(D)) {
      OS << ' ' << ND->getDeclName();
    }

    OS << '\n';
  }
  OS << "DECLS END ---\n";
}
