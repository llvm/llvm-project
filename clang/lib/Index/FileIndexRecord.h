//===--- FileIndexRecord.h - Index data per file --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INDEX_FILEINDEXRECORD_H
#define LLVM_CLANG_LIB_INDEX_FILEINDEXRECORD_H

#include "clang/Index/IndexSymbol.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace clang {
  class IdentifierInfo;

namespace index {

class FileIndexRecord {
public:
  struct DeclOccurrence {
    SymbolRoleSet Roles;
    unsigned Offset;
    const Decl *Dcl;
    SmallVector<SymbolRelation, 3> Relations;

    DeclOccurrence(SymbolRoleSet R,
                   unsigned Offset,
                   const Decl *D,
                   ArrayRef<SymbolRelation> Relations)
      : Roles(R),
        Offset(Offset),
        Dcl(D),
        Relations(Relations.begin(), Relations.end()) {}

    friend bool operator <(const DeclOccurrence &LHS, const DeclOccurrence &RHS) {
      return LHS.Offset < RHS.Offset;
    }
  };

private:
  FileID FID;
  bool IsSystem;
  std::vector<DeclOccurrence> Decls;

public:
  FileIndexRecord(FileID FID, bool isSystem) : FID(FID), IsSystem(isSystem) {}

  ArrayRef<DeclOccurrence> getDeclOccurrences() const { return Decls; }

  FileID getFileID() const { return FID; }
  bool isSystem() const { return IsSystem; }

  void addDeclOccurence(SymbolRoleSet Roles,
                        unsigned Offset,
                        const Decl *D,
                        ArrayRef<SymbolRelation> Relations);
  void print(llvm::raw_ostream &OS);
};

} // end namespace index
} // end namespace clang

#endif
