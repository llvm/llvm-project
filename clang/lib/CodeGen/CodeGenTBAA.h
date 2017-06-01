//===--- CodeGenTBAA.h - TBAA information for LLVM CodeGen ------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is the code that manages TBAA information and defines the TBAA policy
// for the optimizer to use.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CODEGEN_CODEGENTBAA_H
#define LLVM_CLANG_LIB_CODEGEN_CODEGENTBAA_H

#include "clang/AST/Type.h"
#include "clang/Basic/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"

namespace clang {
  class ASTContext;
  class CodeGenOptions;
  class LangOptions;
  class MangleContext;
  class QualType;
  class Type;

namespace CodeGen {
  class CGRecordLayout;

  struct TBAAPathTag {
    TBAAPathTag(const Type *B, const llvm::MDNode *A, uint64_t O)
      : BaseT(B), AccessN(A), Offset(O) {}
    const Type *BaseT;
    const llvm::MDNode *AccessN;
    uint64_t Offset;
  };

/// CodeGenTBAA - This class organizes the cross-module state that is used
/// while lowering AST types to LLVM types.
class CodeGenTBAA {
  ASTContext &Context;
  const CodeGenOptions &CodeGenOpts;
  const LangOptions &Features;
  MangleContext &MContext;

  // MDHelper - Helper for creating metadata.
  llvm::MDBuilder MDHelper;

  /// MetadataCache - This maps clang::Types to scalar llvm::MDNodes describing
  /// them.
  llvm::DenseMap<const Type *, llvm::MDNode *> MetadataCache;
  /// This maps clang::Types to a struct node in the type DAG.
  llvm::DenseMap<const Type *, llvm::MDNode *> StructTypeMetadataCache;
  /// This maps TBAAPathTags to a tag node.
  llvm::DenseMap<TBAAPathTag, llvm::MDNode *> StructTagMetadataCache;
  /// This maps a scalar type to a scalar tag node.
  llvm::DenseMap<const llvm::MDNode *, llvm::MDNode *> ScalarTagMetadataCache;

  /// StructMetadataCache - This maps clang::Types to llvm::MDNodes describing
  /// them for struct assignments.
  llvm::DenseMap<const Type *, llvm::MDNode *> StructMetadataCache;

  llvm::MDNode *Root;
  llvm::MDNode *Char;

  /// getRoot - This is the mdnode for the root of the metadata type graph
  /// for this translation unit.
  llvm::MDNode *getRoot();

  /// getChar - This is the mdnode for "char", which is special, and any types
  /// considered to be equivalent to it.
  llvm::MDNode *getChar();

  /// CollectFields - Collect information about the fields of a type for
  /// !tbaa.struct metadata formation. Return false for an unsupported type.
  bool CollectFields(uint64_t BaseOffset,
                     QualType Ty,
                     SmallVectorImpl<llvm::MDBuilder::TBAAStructField> &Fields,
                     bool MayAlias);

  /// A wrapper function to create a scalar type. For struct-path aware TBAA,
  /// the scalar type has the same format as the struct type: name, offset,
  /// pointer to another node in the type DAG.
  llvm::MDNode *createTBAAScalarType(StringRef Name, llvm::MDNode *Parent);

public:
  CodeGenTBAA(ASTContext &Ctx, llvm::LLVMContext &VMContext,
              const CodeGenOptions &CGO,
              const LangOptions &Features,
              MangleContext &MContext);
  ~CodeGenTBAA();

  /// getTBAAInfo - Get the TBAA MDNode to be used for a dereference
  /// of the given type.
  llvm::MDNode *getTBAAInfo(QualType QTy);

  /// getTBAAInfoForVTablePtr - Get the TBAA MDNode to be used for a
  /// dereference of a vtable pointer.
  llvm::MDNode *getTBAAInfoForVTablePtr();

  /// getTBAAStructInfo - Get the TBAAStruct MDNode to be used for a memcpy of
  /// the given type.
  llvm::MDNode *getTBAAStructInfo(QualType QTy);

  /// Get the MDNode in the type DAG for given struct type QType.
  llvm::MDNode *getTBAAStructTypeInfo(QualType QType);
  /// Get the tag MDNode for a given base type, the actual scalar access MDNode
  /// and offset into the base type.
  llvm::MDNode *getTBAAStructTagInfo(QualType BaseQType,
                                     llvm::MDNode *AccessNode, uint64_t Offset);

  /// Get the scalar tag MDNode for a given scalar type.
  llvm::MDNode *getTBAAScalarTagInfo(llvm::MDNode *AccessNode);
};

}  // end namespace CodeGen
}  // end namespace clang

namespace llvm {

template<> struct DenseMapInfo<clang::CodeGen::TBAAPathTag> {
  static clang::CodeGen::TBAAPathTag getEmptyKey() {
    return clang::CodeGen::TBAAPathTag(
      DenseMapInfo<const clang::Type *>::getEmptyKey(),
      DenseMapInfo<const MDNode *>::getEmptyKey(),
      DenseMapInfo<uint64_t>::getEmptyKey());
  }

  static clang::CodeGen::TBAAPathTag getTombstoneKey() {
    return clang::CodeGen::TBAAPathTag(
      DenseMapInfo<const clang::Type *>::getTombstoneKey(),
      DenseMapInfo<const MDNode *>::getTombstoneKey(),
      DenseMapInfo<uint64_t>::getTombstoneKey());
  }

  static unsigned getHashValue(const clang::CodeGen::TBAAPathTag &Val) {
    return DenseMapInfo<const clang::Type *>::getHashValue(Val.BaseT) ^
           DenseMapInfo<const MDNode *>::getHashValue(Val.AccessN) ^
           DenseMapInfo<uint64_t>::getHashValue(Val.Offset);
  }

  static bool isEqual(const clang::CodeGen::TBAAPathTag &LHS,
                      const clang::CodeGen::TBAAPathTag &RHS) {
    return LHS.BaseT == RHS.BaseT &&
           LHS.AccessN == RHS.AccessN &&
           LHS.Offset == RHS.Offset;
  }
};

}  // end namespace llvm

#endif
