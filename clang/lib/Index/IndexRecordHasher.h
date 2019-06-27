//===--- IndexRecordHasher.h - Index record hashing -----------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_INDEX_INDEXRECORDHASHER_H
#define LLVM_CLANG_LIB_INDEX_INDEXRECORDHASHER_H

#include "clang/Basic/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/Hashing.h"

namespace clang {
  class ASTContext;
  class Decl;
  class DeclarationName;
  class NestedNameSpecifier;
  class QualType;
  class Type;
  template <typename> class CanQual;
  typedef CanQual<Type> CanQualType;

namespace index {
  class FileIndexRecord;

class IndexRecordHasher {
  ASTContext &Ctx;
  llvm::DenseMap<const void *, llvm::hash_code> HashByPtr;

public:
  explicit IndexRecordHasher(ASTContext &Ctx) : Ctx(Ctx) {}
  ASTContext &getASTContext() { return Ctx; }

  llvm::hash_code hashRecord(const FileIndexRecord &Record);
  llvm::hash_code hash(const Decl *D);
  llvm::hash_code hash(QualType Ty);
  llvm::hash_code hash(CanQualType Ty);
  llvm::hash_code hash(DeclarationName Name);
  llvm::hash_code hash(const NestedNameSpecifier *NNS);

private:
  template <typename T>
  llvm::hash_code tryCache(const void *Ptr, T Obj);

  llvm::hash_code hashImpl(const Decl *D);
  llvm::hash_code hashImpl(CanQualType Ty);
  llvm::hash_code hashImpl(DeclarationName Name);
  llvm::hash_code hashImpl(const NestedNameSpecifier *NNS);
};

} // end namespace index
} // end namespace clang

#endif
