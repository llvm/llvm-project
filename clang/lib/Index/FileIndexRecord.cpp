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

static void addOccurrence(std::vector<DeclOccurrence> &Decls,
                          DeclOccurrence Info) {
  auto IsNextOccurence = [&]() -> bool {
    if (Decls.empty())
      return true;
    auto &Last = Decls.back();
    return Last.Offset < Info.Offset;
  };

  if (IsNextOccurence()) {
    Decls.push_back(std::move(Info));
    return;
  }

  // We keep Decls in order as we need to access them in this order in all cases.
  auto It = llvm::upper_bound(Decls, Info);
  Decls.insert(It, std::move(Info));
}

void FileIndexRecord::addDeclOccurence(SymbolRoleSet Roles, unsigned Offset,
                                       const Decl *D,
                                       ArrayRef<SymbolRelation> Relations) {
  assert(D->isCanonicalDecl() &&
         "Occurrences should be associated with their canonical decl");
  addOccurrence(Decls, DeclOccurrence(Roles, Offset, D, Relations));
}

void FileIndexRecord::addMacroOccurence(SymbolRoleSet Roles, unsigned Offset,
                                        const IdentifierInfo *Name,
                                        const MacroInfo *MI) {
  addOccurrence(Decls, DeclOccurrence(Roles, Offset, Name, MI));
}

void FileIndexRecord::removeHeaderGuardMacros() {
  auto It =
      std::remove_if(Decls.begin(), Decls.end(), [](const DeclOccurrence &D) {
        if (const auto *MI = D.DeclOrMacro.dyn_cast<const MacroInfo *>())
          return MI->isUsedForHeaderGuard();
        return false;
      });
  Decls.erase(It, Decls.end());
}

void FileIndexRecord::print(llvm::raw_ostream &OS, SourceManager &SM) const {
  OS << "DECLS BEGIN ---\n";
  for (auto &DclInfo : Decls) {
    if (const auto *D = DclInfo.DeclOrMacro.dyn_cast<const Decl *>()) {
      SourceLocation Loc = SM.getFileLoc(D->getLocation());
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      OS << llvm::sys::path::filename(PLoc.getFilename()) << ':'
         << PLoc.getLine() << ':' << PLoc.getColumn();

      if (const auto *ND = dyn_cast<NamedDecl>(D)) {
        OS << ' ' << ND->getDeclName();
      }
    } else {
      const auto *MI = DclInfo.DeclOrMacro.get<const MacroInfo *>();
      SourceLocation Loc = SM.getFileLoc(MI->getDefinitionLoc());
      PresumedLoc PLoc = SM.getPresumedLoc(Loc);
      OS << llvm::sys::path::filename(PLoc.getFilename()) << ':'
         << PLoc.getLine() << ':' << PLoc.getColumn();
      OS << ' ' << DclInfo.MacroName->getName();
    }

    OS << '\n';
  }
  OS << "DECLS END ---\n";
}
