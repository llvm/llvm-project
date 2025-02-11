//===--- CIRGenTBAA.h - TBAA information for LLVM CIRGen --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the code that manages TBAA information and defines the TBAA policy
// for the optimizer to use.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENTBAA_H
#define LLVM_CLANG_LIB_CIR_CODEGEN_CIRGENTBAA_H
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "clang/AST/Type.h"
#include "clang/Basic/CodeGenOptions.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
namespace clang::CIRGen {
class CIRGenTypes;
enum class TBAAAccessKind : unsigned {
  Ordinary,
  MayAlias,
  Incomplete,
};
// Describes a memory access in terms of TBAA.
struct TBAAAccessInfo {
  TBAAAccessInfo(TBAAAccessKind kind, cir::TBAAAttr baseType,
                 cir::TBAAAttr accessType, uint64_t offset, uint64_t size)
      : kind(kind), baseType(baseType), accessType(accessType), offset(offset) {
  }

  TBAAAccessInfo(cir::TBAAAttr baseType, cir::TBAAAttr accessType,
                 uint64_t offset, uint64_t size)
      : kind(TBAAAccessKind::Ordinary), baseType(baseType),
        accessType(accessType), offset(offset) {}

  explicit TBAAAccessInfo(cir::TBAAAttr accessType, uint64_t size)
      : TBAAAccessInfo(TBAAAccessKind::Ordinary, /* baseType= */ {}, accessType,
                       /* offset= */ 0, size) {}

  TBAAAccessInfo()
      : TBAAAccessInfo(/* accessType= */ nullptr, /* size= */ 0) {};

  static TBAAAccessInfo getMayAliasInfo() {
    return TBAAAccessInfo(TBAAAccessKind::MayAlias, /* baseType= */ {},
                          /* accessType= */ nullptr,
                          /* offset= */ 0, /* size= */ 0);
  }

  bool isMayAlias() const { return kind == TBAAAccessKind::MayAlias; }

  static TBAAAccessInfo getIncompleteInfo() {
    return TBAAAccessInfo(TBAAAccessKind::Incomplete, /* baseType= */ {},
                          /* accessType= */ {},
                          /* offset= */ 0, /* size= */ 0);
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

  /// The base/leading access type. May be null if this access
  /// descriptor represents an access that is not considered to be an access
  /// to an aggregate or union member.
  cir::TBAAAttr baseType;

  /// The final access type. May be null if there is no TBAA
  /// information available about this access.
  cir::TBAAAttr accessType;

  /// The byte offset of the final access within the base one. Must be
  /// zero if the base access type is not specified.
  uint64_t offset;

  /// The size of access, in bytes.
  uint64_t size;
};

/// This class organizes the cross-module state that is used while lowering AST
/// types to LLVM types.
class CIRGenTBAA {
  mlir::MLIRContext *mlirContext;
  [[maybe_unused]] clang::ASTContext &astContext;
  [[maybe_unused]] CIRGenTypes &types;
  mlir::ModuleOp moduleOp;
  [[maybe_unused]] const clang::CodeGenOptions &codeGenOpts;
  [[maybe_unused]] const clang::LangOptions &features;

  llvm::DenseMap<const Type *, cir::TBAAAttr> metadataCache;

  cir::TBAAAttr getChar();

  // An internal helper function to generate metadata used
  // to describe accesses to objects of the given type.
  cir::TBAAAttr getTypeInfoHelper(clang::QualType qty);
  cir::TBAAAttr getScalarTypeInfo(clang::QualType qty);

public:
  CIRGenTBAA(mlir::MLIRContext *mlirContext, clang::ASTContext &astContext,
             CIRGenTypes &types, mlir::ModuleOp moduleOp,
             const clang::CodeGenOptions &codeGenOpts,
             const clang::LangOptions &features);

  /// Get attribute used to describe accesses to objects of the given type.
  cir::TBAAAttr getTypeInfo(clang::QualType qty);

  /// Get TBAA information that describes an access to an object of the given
  /// type.
  TBAAAccessInfo getAccessInfo(clang::QualType accessType);

  /// Get the TBAA information that describes an access to a virtual table
  /// pointer.
  TBAAAccessInfo getVTablePtrAccessInfo(mlir::Type vtablePtrType);

  /// Get the TBAAStruct attributes to be used for a memcpy of the given type.
  mlir::ArrayAttr getTBAAStructInfo(clang::QualType qty);

  /// Get attribute that describes the given base access type. Return null if
  /// the type is not suitable for use in TBAA access tags.
  cir::TBAAAttr getBaseTypeInfo(clang::QualType qty);

  /// Get TBAA tag for a given memory access.
  cir::TBAAAttr getAccessTagInfo(TBAAAccessInfo tbaaInfo);

  /// Get merged TBAA information for the purpose of type casts.
  TBAAAccessInfo mergeTBAAInfoForCast(TBAAAccessInfo sourceInfo,
                                      TBAAAccessInfo targetInfo);

  /// Get merged TBAA information for the purpose of conditional operator.
  TBAAAccessInfo mergeTBAAInfoForConditionalOperator(TBAAAccessInfo infoA,
                                                     TBAAAccessInfo infoB);

  /// Get merged TBAA information for the purpose of memory transfer calls.
  TBAAAccessInfo mergeTBAAInfoForMemoryTransfer(TBAAAccessInfo destInfo,
                                                TBAAAccessInfo srcInfo);
};
} // namespace clang::CIRGen
namespace llvm {
template <> struct DenseMapInfo<clang::CIRGen::TBAAAccessInfo> {
  static clang::CIRGen::TBAAAccessInfo getEmptyKey() {
    unsigned unsignedKey = DenseMapInfo<unsigned>::getEmptyKey();
    return clang::CIRGen::TBAAAccessInfo(
        static_cast<clang::CIRGen::TBAAAccessKind>(unsignedKey),
        DenseMapInfo<cir::TBAAAttr>::getEmptyKey(),
        DenseMapInfo<cir::TBAAAttr>::getEmptyKey(),
        DenseMapInfo<uint64_t>::getEmptyKey(),
        DenseMapInfo<uint64_t>::getEmptyKey());
  }
  static clang::CIRGen::TBAAAccessInfo getTombstoneKey() {
    unsigned unsignedKey = DenseMapInfo<unsigned>::getTombstoneKey();
    return clang::CIRGen::TBAAAccessInfo(
        static_cast<clang::CIRGen::TBAAAccessKind>(unsignedKey),
        DenseMapInfo<cir::TBAAAttr>::getTombstoneKey(),
        DenseMapInfo<cir::TBAAAttr>::getTombstoneKey(),
        DenseMapInfo<uint64_t>::getTombstoneKey(),
        DenseMapInfo<uint64_t>::getTombstoneKey());
  }
  static unsigned getHashValue(const clang::CIRGen::TBAAAccessInfo &val) {
    auto kindValue = static_cast<unsigned>(val.kind);
    return DenseMapInfo<unsigned>::getHashValue(kindValue) ^
           DenseMapInfo<cir::TBAAAttr>::getHashValue(val.baseType) ^
           DenseMapInfo<cir::TBAAAttr>::getHashValue(val.accessType) ^
           DenseMapInfo<uint64_t>::getHashValue(val.offset) ^
           DenseMapInfo<uint64_t>::getHashValue(val.size);
  }
  static bool isEqual(const clang::CIRGen::TBAAAccessInfo &lhs,
                      const clang::CIRGen::TBAAAccessInfo &rhs) {
    return lhs == rhs;
  }
};
} // namespace llvm
#endif
