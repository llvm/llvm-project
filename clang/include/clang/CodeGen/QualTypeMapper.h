//==---- QualTypeMapper.h - Maps Clang QualType to LLVMABI Types -----------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Maps Clang QualType instances to corresponding LLVM ABI type
/// representations. This mapper translates high-level type information from the
/// AST into low-level ABI-specific types that encode size, alignment, and
/// layout details required for code generation and cross-language
/// interoperability.
///
//===----------------------------------------------------------------------===//
#ifndef CLANG_CODEGEN_QUALTYPE_MAPPER_H
#define CLANG_CODEGEN_QUALTYPE_MAPPER_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Type.h"
#include "clang/AST/TypeOrdering.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Allocator.h"

namespace clang {
namespace CodeGen {

class QualTypeMapper {
private:
  clang::ASTContext &ASTCtx;
  llvm::abi::TypeBuilder Builder;

  llvm::DenseMap<clang::QualType, const llvm::abi::Type *> TypeCache;

  const llvm::abi::Type *convertBuiltinType(const clang::BuiltinType *BT,
                                            bool InMemory = false);
  const llvm::abi::Type *convertPointerType(const clang::PointerType *PT);
  const llvm::abi::Type *convertArrayType(const clang::ArrayType *AT);
  const llvm::abi::Type *convertVectorType(const clang::VectorType *VT);
  const llvm::abi::Type *convertRecordType(const clang::RecordType *RT);
  const llvm::abi::Type *convertEnumType(const clang::EnumType *ET);
  const llvm::abi::Type *convertReferenceType(const ReferenceType *RT);
  const llvm::abi::Type *convertComplexType(const ComplexType *CT);
  const llvm::abi::Type *
  convertMemberPointerType(const clang::MemberPointerType *MPT);
  const llvm::abi::Type *convertMatrixType(const ConstantMatrixType *MT);

  const llvm::abi::StructType *convertStructType(const clang::RecordDecl *RD);
  const llvm::abi::StructType *convertUnionType(const clang::RecordDecl *RD,
                                                bool isTransparent = false);
  const llvm::abi::Type *createPointerTypeForPointee(QualType PointeeType);
  const llvm::abi::StructType *convertCXXRecordType(const CXXRecordDecl *RD,
                                                    bool canPassInRegs);

  void computeFieldInfo(const clang::RecordDecl *RD,
                        SmallVectorImpl<llvm::abi::FieldInfo> &Fields,
                        const clang::ASTRecordLayout &Layout);

  llvm::TypeSize getTypeSize(clang::QualType QT) const;
  llvm::Align getTypeAlign(clang::QualType QT) const;
  llvm::Align getPreferredTypeAlign(QualType QT) const;
  uint64_t getPointerSize() const;
  uint64_t getPointerAlign() const;

public:
  explicit QualTypeMapper(clang::ASTContext &Ctx, llvm::BumpPtrAllocator &Alloc)
      : ASTCtx(Ctx), Builder(Alloc) {}

  const llvm::abi::Type *convertType(clang::QualType QT, bool InMemory = false);

  void clearCache() { TypeCache.clear(); }

  llvm::abi::TypeBuilder getTypeBuilder() { return Builder; }
};

} // namespace CodeGen
} // namespace clang

#endif // CLANG_CODEGEN_QUALTYPE_MAPPER_H
