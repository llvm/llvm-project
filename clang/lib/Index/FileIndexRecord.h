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

#include "clang/Basic/SourceLocation.h"
#include "clang/Index/DeclOccurrence.h"
#include "clang/Index/IndexSymbol.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <vector>

namespace clang {
class IdentifierInfo;

namespace index {

/// Stores the declaration occurrences seen in a particular source or header
/// file of a translation unit
class FileIndexRecord {
private:
  FileID FID;
  bool IsSystem;
  std::vector<DeclOccurrence> Decls;

public:
  FileIndexRecord(FileID FID, bool IsSystem) : FID(FID), IsSystem(IsSystem) {}

  ArrayRef<DeclOccurrence> getDeclOccurrencesSortedByOffset() const { return Decls; }

  FileID getFileID() const { return FID; }
  bool isSystem() const { return IsSystem; }

  /// Adds an occurrence of the canonical declaration \c D at the supplied
  /// \c Offset
  ///
  /// \param Roles the roles the occurrence fulfills in this position.
  /// \param Offset the offset in the file of this occurrence.
  /// \param D the canonical declaration this is an occurrence of.
  /// \param Relations the set of symbols related to this occurrence.
  void addDeclOccurence(SymbolRoleSet Roles, unsigned Offset, const Decl *D,
                        ArrayRef<SymbolRelation> Relations);
  void print(llvm::raw_ostream &OS) const;
};

} // end namespace index
} // end namespace clang

#endif
