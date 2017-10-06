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

// TBAAAccessInfo - Describes a memory access in terms of TBAA.
struct TBAAAccessInfo {
  TBAAAccessInfo(QualType BaseType, llvm::MDNode *AccessType, uint64_t Offset)
    : BaseType(BaseType), AccessType(AccessType), Offset(Offset)
  {}

  explicit TBAAAccessInfo(llvm::MDNode *AccessType)
    : TBAAAccessInfo(/* BaseType= */ QualType(), AccessType, /* Offset= */ 0)
  {}

  TBAAAccessInfo()
    : TBAAAccessInfo(/* AccessType= */ nullptr)
  {}

  /// BaseType - The base/leading access type. May be null if this access
  /// descriptor represents an access that is not considered to be an access
  /// to an aggregate or union member.
  QualType BaseType;

  /// AccessType - The final access type. May be null if there is no TBAA
  /// information available about this access.
  llvm::MDNode *AccessType;

  /// Offset - The byte offset of the final access within the base one. Must be
  /// zero if the base access type is not specified.
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

  /// getTypeInfo - Get metadata used to describe accesses to objects of the
  /// given type.
  llvm::MDNode *getTypeInfo(QualType QTy);

  /// getVTablePtrAccessInfo - Get the TBAA information that describes an
  /// access to a virtual table pointer.
  TBAAAccessInfo getVTablePtrAccessInfo();

  /// getTBAAStructInfo - Get the TBAAStruct MDNode to be used for a memcpy of
  /// the given type.
  llvm::MDNode *getTBAAStructInfo(QualType QTy);

  /// getBaseTypeInfo - Get metadata node for a given base access type.
  llvm::MDNode *getBaseTypeInfo(QualType QType);

  /// getAccessTagInfo - Get TBAA tag for a given memory access.
  llvm::MDNode *getAccessTagInfo(TBAAAccessInfo Info);

  /// getMayAliasAccessInfo - Get TBAA information that represents may-alias
  /// accesses.
  TBAAAccessInfo getMayAliasAccessInfo();
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
