//===--- CIRGenTBAA.h - TBAA information for CIR CodeGen --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the code that manages TBAA information and defines the TBAA policy
// for the optimizer to use. It mirrors clang/lib/CodeGen/CodeGenTBAA.h, but
// produces CIR TBAA attributes instead of LLVM metadata.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_LIB_CIR_CODEGEN_CIRGENTBAA_H
#define CLANG_LIB_CIR_CODEGEN_CIRGENTBAA_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/AST/Type.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include <memory>

namespace clang {
class ASTContext;
class CodeGenOptions;
class LangOptions;
class MangleContext;
} // namespace clang

namespace clang::CIRGen {
class CIRGenCXXABI;

enum class TBAAAccessKind : unsigned {
  Ordinary,
  MayAlias,
  Incomplete,
};

/// Describes a memory access in terms of TBAA.
struct TBAAAccessInfo {
  TBAAAccessInfo(TBAAAccessKind kind, mlir::Attribute baseType,
                 mlir::Attribute accessType, uint64_t offset, uint64_t size)
      : kind(kind), baseType(baseType), accessType(accessType), offset(offset),
        size(size) {}

  TBAAAccessInfo(mlir::Attribute baseType, mlir::Attribute accessType,
                 uint64_t offset, uint64_t size)
      : TBAAAccessInfo(TBAAAccessKind::Ordinary, baseType, accessType, offset,
                       size) {}

  explicit TBAAAccessInfo(mlir::Attribute accessType, uint64_t size)
      : TBAAAccessInfo(/*baseType=*/{}, accessType, /*offset=*/0, size) {}

  TBAAAccessInfo() : TBAAAccessInfo(/*accessType=*/{}, /*size=*/0) {}

  static TBAAAccessInfo getMayAliasInfo() {
    return TBAAAccessInfo(TBAAAccessKind::MayAlias, /*baseType=*/{},
                          /*accessType=*/{}, /*offset=*/0, /*size=*/0);
  }

  bool isMayAlias() const { return kind == TBAAAccessKind::MayAlias; }

  static TBAAAccessInfo getIncompleteInfo() {
    return TBAAAccessInfo(TBAAAccessKind::Incomplete, /*baseType=*/{},
                          /*accessType=*/{}, /*offset=*/0, /*size=*/0);
  }

  bool isIncomplete() const { return kind == TBAAAccessKind::Incomplete; }

  bool operator==(const TBAAAccessInfo &other) const {
    return kind == other.kind && baseType == other.baseType &&
           accessType == other.accessType && offset == other.offset &&
           size == other.size;
  }

  bool operator!=(const TBAAAccessInfo &other) const {
    return !(*this == other);
  }

  explicit operator bool() const { return *this != TBAAAccessInfo(); }

  /// The kind of the access descriptor.
  TBAAAccessKind kind;

  /// The base/leading access type. May be null if this access descriptor
  /// represents an access that is not considered to be an access to an
  /// aggregate or union member.
  mlir::Attribute baseType;

  /// The final access type. May be null if there is no TBAA information
  /// available about this access.
  mlir::Attribute accessType;

  /// The byte offset of the final access within the base one. Must be zero if
  /// the base access type is not specified.
  uint64_t offset;

  /// The size of access, in bytes.
  uint64_t size;
};

/// This class organizes the cross-module state that is used while generating
/// TBAA attributes for memory accesses.
class CIRGenTBAA {
  mlir::MLIRContext &mlirContext;
  clang::ASTContext &astContext;
  CIRGenCXXABI &cxxABI;
  const clang::CodeGenOptions &codeGenOpts;
  const clang::LangOptions &features;
  std::unique_ptr<clang::MangleContext> mangleCtx;

  /// Maps clang types to the type descriptors used to describe accesses to
  /// objects of these types.
  llvm::DenseMap<const clang::Type *, mlir::Attribute> metadataCache;
  /// Maps clang types to their base access type descriptors.
  llvm::DenseMap<const clang::Type *, mlir::Attribute> baseTypeMetadataCache;

  /// The "any pointer" type descriptors, indexed by pointer depth minus one.
  llvm::SmallVector<cir::TBAAScalarAttr, 4> anyPtrs;

  /// Returns the type descriptor of the omnipotent char type.
  cir::TBAAOmnipotentCharAttr getChar();

  /// Returns the type descriptor of the "any" pointer of depth \p ptrDepth.
  cir::TBAAScalarAttr getAnyPtr(unsigned ptrDepth = 1);

  /// Creates a scalar type descriptor; a null \p parent means omnipotent char.
  cir::TBAAScalarAttr createScalarTypeNode(llvm::StringRef name,
                                           cir::TBAAScalarAttr parent,
                                           uint64_t size);

  /// Returns the type descriptor of the given type, without caching.
  mlir::Attribute getTypeInfoHelper(const clang::Type *ty);

  /// Returns the base type descriptor of the given valid base type.
  mlir::Attribute getValidBaseTypeInfo(clang::QualType qty);

  /// Returns the base type descriptor of the given type, without caching.
  mlir::Attribute getBaseTypeInfoHelper(const clang::Type *ty);

public:
  CIRGenTBAA(mlir::MLIRContext &mlirContext, clang::ASTContext &astContext,
             CIRGenCXXABI &cxxABI, const clang::CodeGenOptions &codeGenOpts,
             const clang::LangOptions &features);
  ~CIRGenTBAA();

  /// Returns the type descriptor used to describe accesses to objects of the
  /// given type.
  mlir::Attribute getTypeInfo(clang::QualType qty);

  /// Returns the TBAA information that describes an access to an object of the
  /// given type.
  TBAAAccessInfo getAccessInfo(clang::QualType accessType);

  /// Returns the type descriptor that describes the given base access type, or
  /// null if the type is not suitable for use in TBAA access tags.
  mlir::Attribute getBaseTypeInfo(clang::QualType qty);

  /// Returns the access tag for the given memory access, or null if the access
  /// has no TBAA information.
  cir::TBAATagAttr getAccessTagInfo(TBAAAccessInfo info);

  /// Returns the merged TBAA information for the purpose of type casts.
  TBAAAccessInfo mergeTBAAInfoForCast(TBAAAccessInfo sourceInfo,
                                      TBAAAccessInfo targetInfo);

  /// Returns the merged TBAA information for the purpose of conditional
  /// operator.
  TBAAAccessInfo mergeTBAAInfoForConditionalOperator(TBAAAccessInfo infoA,
                                                     TBAAAccessInfo infoB);
};

} // namespace clang::CIRGen

#endif // CLANG_LIB_CIR_CODEGEN_CIRGENTBAA_H
