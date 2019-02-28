//===--- FileIndexRecord.cpp - Index data per file ------------------------===//
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
  auto It = std::upper_bound(Decls.begin(), Decls.end(), NewInfo);
  Decls.insert(It, std::move(NewInfo));
}

void FileIndexRecord::print(llvm::raw_ostream &OS) const {
  OS << "DECLS BEGIN ---\n";
  for (auto &DclInfo : Decls) {
    auto D = DclInfo.Dcl;
    SourceManager &SM = D->getASTContext().getSourceManager();
    SourceLocation Loc = SM.getFileLoc(D->getLocation());
    PresumedLoc PLoc = SM.getPresumedLoc(Loc);
    OS << llvm::sys::path::filename(PLoc.getFilename()) << ':' << PLoc.getLine()
       << ':' << PLoc.getColumn();

    if (auto ND = dyn_cast<NamedDecl>(D)) {
      OS << ' ' << ND->getNameAsString();
    }

    OS << '\n';
  }
  OS << "DECLS END ---\n";
}
